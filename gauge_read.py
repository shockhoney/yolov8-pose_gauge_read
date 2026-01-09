import math
import re
import cv2
import numpy as np
import argparse
import logging
from ultralytics import YOLO
from paddleocr import PaddleOCR

# -----------------------------------------------------------
# 配置与常量
# -----------------------------------------------------------
# 设置日志级别，屏蔽 PaddleOCR 的过多调试信息
logging.getLogger("ppocr").setLevel(logging.WARNING)

# YOLO 类别 ID (根据你的数据集设定)
CLS_CENTER = 0
CLS_GAUGE  = 1
CLS_MAX    = 2
CLS_MIN    = 3
CLS_TIP    = 4

# 初始化 PaddleOCR
# 修正点：
# 1. 去掉 show_log (新版不支持)
# 2. 将 use_angle_cls 改为 use_textline_orientation (新版推荐)
try:
    ocr_engine = PaddleOCR(use_textline_orientation=True, lang='en')
except Exception:
    # 兼容旧版本
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

# -----------------------------------------------------------
# 数学与几何工具函数
# -----------------------------------------------------------
def get_angle(center, point):
    """
    计算点 point 相对于 center 的角度 (0-360度)。
    图像坐标系：X轴向右，Y轴向下。
    0度: 3点钟方向
    90度: 6点钟方向
    """
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360
    return angle

def calculate_value(angle_min, angle_max, angle_tip, val_min, val_max):
    """
    根据角度占比计算当前读数 (假设表盘顺时针增长)
    """
    # 计算量程的总角度跨度 (处理跨越0度轴的情况)
    total_span = (angle_max - angle_min + 360) % 360
    
    # 计算指针相对于最小刻度的角度跨度
    current_span = (angle_tip - angle_min + 360) % 360
    
    # 异常保护：量程跨度太小说明检测有问题
    if total_span < 10: 
        return val_min

    ratio = current_span / total_span
    
    # --- 钳位逻辑 (Clamping) ---
    # 防止指针轻微抖动导致读数跳变 (例如从 0 跳到 最大值)
    # 假设：如果计算出的数值超过最大量程的 20%，则认为是“小于最小值”的情况（即指针在最小刻度左侧）
    if ratio > 1.2:
        ratio = 0.0  # 归零
    elif ratio > 1.0:
        ratio = 1.0  # 满量程封顶
        
    value = val_min + ratio * (val_max - val_min)
    return value

def parse_number(text):
    """从 OCR 文本中提取数字，处理常见误识别"""
    # 替换常见误识字符 (l->1, O->0, ,->.)
    text = text.lower().replace('l', '1').replace('o', '0').replace(',', '.')
    # 匹配浮点数或整数
    match = re.search(r"-?\d+(\.\d+)?", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None

def dist_sq(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

# -----------------------------------------------------------
# 核心处理逻辑
# -----------------------------------------------------------
def process_one_image(model, img_path, out_path, pad=10):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[Error] 无法读取图片: {img_path}")
        return

    # YOLO 推理
    results = model(img, verbose=False)[0]
    boxes = results.boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, cls]

    gauges = boxes[boxes[:, 5] == CLS_GAUGE]
    if len(gauges) == 0:
        print(f"[Info] {img_path}: 未检测到仪表盘 (Gauge)")
        return

    print(f"[Info] 处理图片: {img_path}, 检测到 {len(gauges)} 个仪表盘")

    for i, g_box in enumerate(gauges):
        gx1, gy1, gx2, gy2 = map(int, g_box[:4])
        
        # 定义辅助函数：在当前 Gauge 框内找置信度最高的指定类别
        def get_best_cls_in_gauge(cls_id):
            c_boxes = boxes[boxes[:, 5] == cls_id]
            candidates = []
            for b in c_boxes:
                cx = (b[0] + b[2]) / 2
                cy = (b[1] + b[3]) / 2
                # 宽松判断：中心点在 Gauge 框内 (或稍微扩大一点范围)
                if (gx1 - pad) < cx < (gx2 + pad) and (gy1 - pad) < cy < (gy2 + pad):
                    candidates.append(b)
            if not candidates: return None
            return max(candidates, key=lambda x: x[4]) # 取置信度最高

        kps = {
            'center': get_best_cls_in_gauge(CLS_CENTER),
            'min':    get_best_cls_in_gauge(CLS_MIN),
            'max':    get_best_cls_in_gauge(CLS_MAX),
            'tip':    get_best_cls_in_gauge(CLS_TIP)
        }

        # 校验关键点完整性
        missing = [k for k, v in kps.items() if v is None]
        if missing:
            print(f"  -> Gauge {i}: 缺少关键点 {missing}，跳过计算")
            cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
            continue

        # --- 1. 获取关键点坐标 ---
        def get_pt(box): return ((box[0]+box[2])/2, (box[1]+box[3])/2)
        pt_center = get_pt(kps['center'])
        pt_min    = get_pt(kps['min'])
        pt_max    = get_pt(kps['max'])
        pt_tip    = get_pt(kps['tip'])

        # --- 2. OCR 识别数值 ---
        # 裁剪 ROI 进行识别，提高准确率
        h, w = img.shape[:2]
        rx1, ry1 = max(0, gx1 - pad), max(0, gy1 - pad)
        rx2, ry2 = min(w, gx2 + pad), min(h, gy2 + pad)
        roi = img[ry1:ry2, rx1:rx2]

        ocr_result = ocr_engine.ocr(roi, cls=True)
        
        ocr_vals = []
        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                text = line[1][0]
                val = parse_number(text)
                if val is not None:
                    # 将 ROI 坐标映射回原图全局坐标
                    box_pts = np.array(line[0])
                    cx_roi = np.mean(box_pts[:, 0])
                    cy_roi = np.mean(box_pts[:, 1])
                    ocr_vals.append({'val': val, 'center': (cx_roi + rx1, cy_roi + ry1)})

        if not ocr_vals:
            print(f"  -> Gauge {i}: OCR 未识别到数字，无法读数")
            continue

        # --- 3. 空间匹配 (Spatial Matching) ---
        # 找离 Min 检测点最近的数字，和离 Max 检测点最近的数字
        val_min = min(ocr_vals, key=lambda x: dist_sq(pt_min, x['center']))['val']
        val_max = min(ocr_vals, key=lambda x: dist_sq(pt_max, x['center']))['val']

        # 简单的逻辑修正: Max 应该大于 Min
        if val_max < val_min:
            print(f"  -> Gauge {i}: [警告] 识别到 Max({val_max}) < Min({val_min})，自动交换")
            val_min, val_max = val_max, val_min

        # --- 4. 计算最终读数 ---
        ang_min = get_angle(pt_center, pt_min)
        ang_max = get_angle(pt_center, pt_max)
        ang_tip = get_angle(pt_center, pt_tip)

        reading = calculate_value(ang_min, ang_max, ang_tip, val_min, val_max)
        
        print(f"  -> Gauge {i}: 读数 = {reading:.3f} (范围: {val_min} ~ {val_max})")

        # --- 5. 绘图可视化 ---
        # 画框
        cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
        # 画连线
        cv2.line(img, (int(pt_center[0]), int(pt_center[1])), (int(pt_tip[0]), int(pt_tip[1])), (0, 0, 255), 2)
        # 写字
        label = f"{reading:.2f}"
        cv2.putText(img, label, (gx1, gy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Range: {val_min}-{val_max}", (gx1, gy2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    cv2.imwrite(out_path, img)
    print(f"[Done] 结果已保存至: {out_path}")

# -----------------------------------------------------------
# 主程序入口
# -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO+PaddleOCR 仪表读数工具")
    parser.add_argument("--weights", type=str, required=True, help="YOLO 模型路径 (例如 best.pt)")
    parser.add_argument("--source", type=str, required=True, help="输入图片路径")
    parser.add_argument("--output", type=str, default="result.jpg", help="输出图片路径")
    
    args = parser.parse_args()
    
    process_one_image(args.weights, args.source, args.output)
