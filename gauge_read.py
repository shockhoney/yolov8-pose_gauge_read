import cv2
import math
import numpy as np
import re
import argparse
import sys
from ultralytics import YOLO

# -----------------------------------------------------------
# 0. 环境初始化
# -----------------------------------------------------------
try:
    import easyocr
except ImportError:
    print("[Error] 请先安装 easyocr: pip install easyocr")
    sys.exit(1)

print("[Init] 初始化 OCR 引擎...")
reader = easyocr.Reader(['en'], gpu=True) # 如无GPU会自动切CPU

# YOLO 类别 ID
CLS_CENTER, CLS_GAUGE, CLS_MAX, CLS_MIN, CLS_TIP = 0, 1, 2, 3, 4

# -----------------------------------------------------------
# 1. 核心读数逻辑 (优化版)
# -----------------------------------------------------------
def calculate_angle(center, point):
    """
    计算点相对于圆心的绝对角度 (0-360度)
    使用 arctan2，返回值范围 (-180, 180]，映射到 [0, 360)
    """
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360
    return angle

def get_reading(pt_c, pt_min, pt_max, pt_tip, vmin, vmax):
    """
    根据角度比例计算读数
    逻辑：将坐标系旋转，使 Min 点变为 0度，计算 Tip 的相对偏移量
    """
    # 1. 计算三个关键点的绝对角度
    ang_min = calculate_angle(pt_c, pt_min)
    ang_max = calculate_angle(pt_c, pt_max)
    ang_tip = calculate_angle(pt_c, pt_tip)

    # 2. 计算相对角度 (顺时针扫过的角度)
    #    公式: (目标角度 - 起始角度 + 360) % 360
    #    这样 Min 点永远是 0 度，避免了跨越 X 轴的数学问题
    span_total = (ang_max - ang_min + 360) % 360  # 量程总角度 (通常约 270度)
    span_tip   = (ang_tip - ang_min + 360) % 360  # 指针当前角度

    # 3. 异常保护：防止 Min/Max 重叠或检测错误
    if span_total < 10: 
        return vmin

    # 4. 读数计算与死区处理
    #    正常情况：指针角度在 0 到 总量程 之间
    if span_tip <= span_total:
        progress = span_tip / span_total
        value = vmin + progress * (vmax - vmin)
        return value
    
    #    死区情况：指针落在 Max 和 Min 之间的空白区
    #    逻辑：看它是离 Max 近，还是离 Min 近
    else:
        # 离 Max 的距离 (正向超出)
        dist_to_max = span_tip - span_total
        # 离 Min 的距离 (反向绕回)
        dist_to_min = 360 - span_tip
        
        if dist_to_min < dist_to_max:
            return vmin # 钳位到最小值
        else:
            return vmax # 钳位到最大值

# -----------------------------------------------------------
# 2. OCR 辅助逻辑
# -----------------------------------------------------------
def parse_number(text):
    """从 OCR 文本中提取数字，处理常见误识别"""
    text = text.replace(',', '.').replace('l', '1').replace('O', '0').lower()
    match = re.search(r"-?\d+(\.\d+)?", text)
    return float(match.group()) if match else None

def get_range_ocr(img, gauge_box, pt_min, pt_max):
    """裁剪 ROI -> 增强 -> 识别 -> 空间匹配"""
    gx1, gy1, gx2, gy2 = map(int, gauge_box)
    h, w = img.shape[:2]
    
    # 裁剪并留一点边距
    pad = 15
    rx1, ry1 = max(0, gx1-pad), max(0, gy1-pad)
    rx2, ry2 = min(w, gx2+pad), min(h, gy2+pad)
    roi = img[ry1:ry2, rx1:rx2]
    if roi.size == 0: return 0.0, 1.6 # 默认值

    # 图像增强：灰度 -> 高斯模糊 -> CLAHE (去反光)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(blur)

    # EasyOCR 识别
    results = reader.readtext(enhanced, detail=1)
    
    candidates = []
    for (bbox, text, prob) in results:
        val = parse_number(text)
        if val is not None and prob > 0.3:
            # 转换回原图坐标
            cx = (bbox[0][0] + bbox[2][0])/2 + rx1
            cy = (bbox[0][1] + bbox[2][1])/2 + ry1
            candidates.append({'val': val, 'center': (cx, cy)})

    if len(candidates) < 2: return 0.0, 1.6 # 识别失败用默认

    # 找离 Min/Max 点最近的数字
    def dist_sq(p1, p2): return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    vmin = min(candidates, key=lambda x: dist_sq(pt_min, x['center']))['val']
    vmax = min(candidates, key=lambda x: dist_sq(pt_max, x['center']))['val']

    if vmax < vmin: vmin, vmax = vmax, vmin
    return vmin, vmax

# -----------------------------------------------------------
# 3. 主程序
# -----------------------------------------------------------
def process_gauge(weights, source, output):
    model = YOLO(weights)
    img = cv2.imread(source)
    if img is None: return

    results = model(img, verbose=False)[0]
    boxes = results.boxes.data.cpu().numpy()
    
    gauges = boxes[boxes[:, 5] == CLS_GAUGE]
    print(f"[Info] 检测到 {len(gauges)} 个仪表盘")

    for i, g_box in enumerate(gauges):
        gx1, gy1, gx2, gy2 = g_box[:4]
        
        # 辅助函数：获取框中心点
        def get_pt(cls_id):
            # 筛选出位于当前 Gauge 框内部的特定类别
            c_boxes = [b for b in boxes[boxes[:, 5] == cls_id] 
                       if (gx1-20) < (b[0]+b[2])/2 < (gx2+20) and 
                          (gy1-20) < (b[1]+b[3])/2 < (gy2+20)]
            if not c_boxes: return None
            best = max(c_boxes, key=lambda x: x[4]) # 取置信度最高
            return ((best[0]+best[2])/2, (best[1]+best[3])/2)

        pt_c   = get_pt(CLS_CENTER)
        pt_min = get_pt(CLS_MIN)
        pt_max = get_pt(CLS_MAX)
        pt_tip = get_pt(CLS_TIP)

        cv2.rectangle(img, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 255, 0), 2)

        if not (pt_c and pt_min and pt_max and pt_tip):
            print(f"  -> Gauge {i+1}: 关键点缺失，无法计算")
            continue

        # 1. 获取量程
        vmin, vmax = get_range_ocr(img, g_box[:4], pt_min, pt_max)
        
        # 2. 计算读数
        value = get_reading(pt_c, pt_min, pt_max, pt_tip, vmin, vmax)
        print(f"  -> Gauge {i+1}: 读数={value:.3f} (量程 {vmin}~{vmax})")

        # 3. 绘图
        cv2.line(img, (int(pt_c[0]), int(pt_c[1])), (int(pt_tip[0]), int(pt_tip[1])), (0, 0, 255), 2)
        cv2.circle(img, (int(pt_min[0]), int(pt_min[1])), 4, (255,0,0), -1) # Min 蓝
        cv2.circle(img, (int(pt_max[0]), int(pt_max[1])), 4, (0,0,255), -1) # Max 红
        
        text = f"{value:.2f}"
        cv2.putText(img, text, (int(gx1), int(gy1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(img, f"Range: {vmin}-{vmax}", (int(gx1), int(gy2)+20), 0, 0.6, (255, 255, 255), 1)

    cv2.imwrite(output, img)
    print(f"[Done] 结果已保存: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="best.pt 路径")
    parser.add_argument("--source", type=str, required=True, help="图片路径")
    parser.add_argument("--output", type=str, default="result.jpg")
    args = parser.parse_args()
    
    process_gauge(args.weights, args.source, args.output)