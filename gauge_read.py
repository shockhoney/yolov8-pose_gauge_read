import math
import re
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

# -----------------------------------------------------------
# 配置与常量
# -----------------------------------------------------------
# YOLO 类别 ID (根据你的模型设定)
CLS_CENTER = 0
CLS_GAUGE  = 1
CLS_MAX    = 2
CLS_MIN    = 3
CLS_TIP    = 4

# 初始化 PaddleOCR (只需初始化一次，自动下载模型)
# use_angle_cls=True 可以纠正文字方向，lang='en' 对数字支持更好
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

# -----------------------------------------------------------
# 数学与几何工具函数
# -----------------------------------------------------------
def get_angle(center, point):
    """
    计算点 point 相对于 center 的角度 (0-360度)。
    图像坐标系：X轴向右，Y轴向下。
    0度: 3点钟方向 (X轴正向)
    90度: 6点钟方向 (Y轴正向)
    """
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    # atan2 返回 (-pi, pi)，转换为角度
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360
    return angle

def calculate_value(angle_min, angle_max, angle_tip, val_min, val_max):
    """
    根据角度占比计算当前读数
    """
    # 计算量程的总角度跨度 (假设顺时针为增长方向)
    # (End - Start + 360) % 360 处理跨越0度轴的情况
    total_span = (angle_max - angle_min + 360) % 360
    
    # 计算指针相对于最小刻度的角度跨度
    current_span = (angle_tip - angle_min + 360) % 360
    
    # 异常情况处理：如果总跨度非常小(模型检测错误)或指针跑偏
    if total_span == 0: 
        return val_min

    # 简单逻辑：如果 current_span > total_span，说明指针在量程外
    # 比如: 量程是 270度，指针转了 300度(超量程) 或 指针在 0度之前(小于最小)
    # 这里做一个简单的钳位逻辑
    ratio = current_span / total_span
    
    # 如果比例略大于1 (如1.05)，可能是超量程；如果接近 360/270，可能是小于最小值
    # 这里设定一个阈值，比如 1.2 倍量程视为“小于最小值”（因为它是转了一大圈）
    if ratio > 1.2:
        # 视为小于最小值 (归0)
        ratio = 0.0
    elif ratio > 1.0:
        # 视为满量程
        ratio = 1.0
        
    value = val_min + ratio * (val_max - val_min)
    return value

def parse_number(text):
    """从 OCR 文本中提取数字"""
    # 替换常见误识 (l->1, O->0)
    text = text.lower().replace('l', '1').replace('o', '0')
    # 匹配浮点数或整数
    match = re.search(r"-?\d+(\.\d+)?", text)
    if match:
        return float(match.group())
    return None

def dist_sq(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

# -----------------------------------------------------------
# 核心处理逻辑
# -----------------------------------------------------------
def process_gauge(img, gauge_box, keypoints, pad=10):
    """
    处理单个表盘区域
    gauge_box: [x1, y1, x2, y2]
    keypoints: dict of best boxes {'center': box, 'min': box, ...}
    """
    gx1, gy1, gx2, gy2 = map(int, gauge_box[:4])
    
    # 1. 提取关键点中心坐标 (全局坐标)
    def get_center(box):
        return ((box[0]+box[2])/2, (box[1]+box[3])/2)
    
    pt_center = get_center(keypoints['center'])
    pt_tip    = get_center(keypoints['tip'])
    pt_min_box_center = get_center(keypoints['min'])
    pt_max_box_center = get_center(keypoints['max'])

    # 2. 裁剪表盘 ROI 用于 OCR
    # 适当padding防止数字被切断
    h, w = img.shape[:2]
    rx1 = max(0, gx1 - pad)
    ry1 = max(0, gy1 - pad)
    rx2 = min(w, gx2 + pad)
    ry2 = min(h, gy2 + pad)
    roi = img[ry1:ry2, rx1:rx2]

    # 3. OCR 识别数字
    result = ocr_engine.ocr(roi, cls=True)
    
    # 提取所有数字及其全局中心点
    ocr_numbers = []
    if result and result[0]:
        for line in result[0]:
            # line: [[[tl, tr, br, bl], (text, conf)]]
            text = line[1][0]
            val = parse_number(text)
            if val is not None:
                # 计算文本框中心 (相对于 ROI)
                box_pts = np.array(line[0])
                cx_roi = np.mean(box_pts[:, 0])
                cy_roi = np.mean(box_pts[:, 1])
                # 转换为全局坐标
                cx_global = cx_roi + rx1
                cy_global = cy_roi + ry1
                ocr_numbers.append({'val': val, 'center': (cx_global, cy_global)})

    if not ocr_numbers:
        print("[Warn] OCR 未识别到任何数字")
        return None

    # 4. 匹配 Min 和 Max 的数值
    # 策略：寻找距离 'min' 检测框中心最近的 OCR 数字
    def find_nearest_val(target_pt, candidates):
        if not candidates: return 0.0
        #按距离排序
        best = min(candidates, key=lambda x: dist_sq(target_pt, x['center']))
        return best['val']

    val_min = find_nearest_val(pt_min_box_center, ocr_numbers)
    val_max = find_nearest_val(pt_max_box_center, ocr_numbers)

    # 简单的逻辑修正：通常 Max > Min，如果反了可能匹配错，或者表盘特殊
    # 如果识别出的 max < min，可能是把负数识别错了，或者位置匹配错误。
    # 这里假设正常逻辑交换
    if val_max < val_min:
        val_min, val_max = val_max, val_min
        
    # 5. 计算角度
    ang_min = get_angle(pt_center, pt_min_box_center)
    ang_max = get_angle(pt_center, pt_max_box_center)
    ang_tip = get_angle(pt_center, pt_tip)

    # 6. 计算最终读数
    result_val = calculate_value(ang_min, ang_max, ang_tip, val_min, val_max)
    
    return {
        'value': result_val,
        'range': (val_min, val_max),
        'angles': (ang_min, ang_max, ang_tip),
        'roi_coords': (rx1, ry1, rx2, ry2)
    }

def main(weights_path, img_path):
    # 加载你的 YOLO 模型
    model = YOLO(weights_path)
    
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not found")
        
    # 推理
    results = model(img)[0]
    boxes = results.boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, cls]

    # 分离各个类别的框
    gauges = boxes[boxes[:, 5] == CLS_GAUGE]
    
    # 遍历每个检测到的仪表盘 (支持一张图多个表)
    for i, g_box in enumerate(gauges):
        # 筛选在这个 gauge 内部的其他关键点
        gx1, gy1, gx2, gy2 = g_box[:4]
        
        # 辅助函数：找该 gauge 内置信度最高的某个类别的框
        def get_best_cls(cls_id):
            # 找到所有该类别的框
            c_boxes = boxes[boxes[:, 5] == cls_id]
            candidates = []
            for b in c_boxes:
                # 判断中心点是否在 gauge 框内
                cx = (b[0] + b[2]) / 2
                cy = (b[1] + b[3]) / 2
                if gx1 < cx < gx2 and gy1 < cy < gy2:
                    candidates.append(b)
            if not candidates: return None
            # 返回置信度最高的
            candidates.sort(key=lambda x: x[4], reverse=True)
            return candidates[0]

        # 获取关键组件
        kps = {
            'center': get_best_cls(CLS_CENTER),
            'min':    get_best_cls(CLS_MIN),
            'max':    get_best_cls(CLS_MAX),
            'tip':    get_best_cls(CLS_TIP)
        }

        # 检查是否所有关键点都齐全
        if any(v is None for v in kps.values()):
            print(f"Gauge {i}: 缺少关键点 (Center/Min/Max/Tip)，跳过。")
            continue

        # 计算读数
        res = process_gauge(img, g_box, kps)
        
        if res:
            val = res['value']
            vmin, vmax = res['range']
            
            # --- 可视化绘图 ---
            # 画 Gauge 框
            cv2.rectangle(img, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 255, 0), 2)
            
            # 显示读数
            text = f"Val: {val:.2f} (Range: {vmin}-{vmax})"
            print(f"Gauge {i} Result: {text}")
            
            # 在图上绘制文字
            label_pos = (int(gx1), int(gy1) - 10)
            cv2.putText(img, text, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # 画连接线示意角度 (可选)
            cx = (kps['center'][0] + kps['center'][2])//2
            cy = (kps['center'][1] + kps['center'][3])//2
            tx = (kps['tip'][0] + kps['tip'][2])//2
            ty = (kps['tip'][1] + kps['tip'][3])//2
            cv2.line(img, (int(cx), int(cy)), (int(tx), int(ty)), (0, 0, 255), 3)

    # 保存结果
    out_path = "result.jpg"
    cv2.imwrite(out_path, img)
    print(f"Result saved to {out_path}")

if __name__ == "__main__":
    WEIGHTS = "/home/devops/works/analog-gauge-reader/runs/detect/train4/weights/best.pt"  
    IMAGE   = "test.jpg" 
    
    try:
        main(WEIGHTS, IMAGE)
    except Exception as e:
        print(f"Error: {e}")
