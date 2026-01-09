import cv2
import math
import numpy as np
import re
import argparse
import sys
from ultralytics import YOLO

# -----------------------------------------------------------
# 0. 环境检查
# -----------------------------------------------------------
try:
    import easyocr
except ImportError:
    print("[Error] 缺少 easyocr。请运行: pip install easyocr")
    sys.exit(1)

# 初始化 EasyOCR
print("[Init] 初始化 OCR...")
reader = easyocr.Reader(['en'], gpu=True) 

# YOLO 类别
CLS_CENTER = 0
CLS_GAUGE  = 1
CLS_MAX    = 2
CLS_MIN    = 3
CLS_TIP    = 4

# -----------------------------------------------------------
# 优化后的数学逻辑
# -----------------------------------------------------------
def get_angle_vector(center, point):
    """
    计算向量 (center -> point) 的绝对角度 (0-360)
    X轴正向为0度，Y轴向下为90度 (图像坐标系)
    """
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    # atan2 返回范围 [-180, 180]
    angle = math.degrees(math.atan2(dy, dx))
    # 转换为 [0, 360)
    if angle < 0:
        angle += 360
    return angle

def calculate_reading_optimized(pt_c, pt_min, pt_max, pt_tip, vmin, vmax):
    """
    【核心优化版】读数计算逻辑
    """
    # 1. 获取三个关键点的绝对角度
    ang_min = get_angle_vector(pt_c, pt_min)
    ang_max = get_angle_vector(pt_c, pt_max)
    ang_tip = get_angle_vector(pt_c, pt_tip)

    # 2. 角度归一化 (将坐标系旋转，使 Min 点变为 0 度)
    #    这样所有的角度都是“相对于 Min 点的顺时针偏移量”
    #    公式: (Angle - Min_Angle + 360) % 360
    
    # 量程的总跨度 (Min -> Max)
    span_total = (ang_max - ang_min + 360) % 360
    
    # 指针的当前跨度 (Min -> Tip)
    span_tip   = (ang_tip - ang_min + 360) % 360

    # -------------------------------------------------------
    # 3. 异常与死区处理 (Optimization Logic)
    # -------------------------------------------------------
    
    # 情况 A: 错误的 Min/Max 检测 (量程角度过小或过大)
    # 一般表盘有效量程在 90度 ~ 320度 之间
    if span_total < 10:
        print(f"  [Warn] 量程跨度异常 ({span_total:.1f}°)，可能 Min/Max 重叠")
        return vmin
    
    # 情况 B: 指针在量程范围内 (正常情况)
    if span_tip <= span_total:
        progress = span_tip / span_total
        return vmin + progress * (vmax - vmin)
    
    # 情况 C: 指针在“死区” (Dead Zone) -> 即 Max 和 Min 之间的空白区域
    # 此时 span_tip 会大于 span_total。我们需要判断它是“低于最小值”还是“高于最大值”。
    else:
        # 计算指针距离 Min 更近，还是距离 Max 更近
        # 距离 Min 的反向距离 (即 360 - span_tip)
        dist_to_min = 360 - span_tip
        # 距离 Max 的正向距离
        dist_to_max = span_tip - span_total
        
        if dist_to_min < dist_to_max:
            # 指针在 Min 的左边 (略小于 0)
            # 如果你想允许负读数 (如 -0.1)，可以放开下面这行：
            # return vmin - (dist_to_min / span_total) * (vmax - vmin)
            return vmin # 钳位到最小值
        else:
            # 指针在 Max 的右边 (爆表)
            return vmax # 钳位到最大值

# -----------------------------------------------------------
# OCR 与 辅助函数
# -----------------------------------------------------------
def dist_sq(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def parse_number(text):
    text = text.replace(',', '.').replace('l', '1').replace('O', '0').lower()
    match = re.search(r"-?\d+(\.\d+)?", text)
    if match:
        try: return float(match.group())
        except: return None
    return None

def get_range_values(img, gauge_box, pt_min, pt_max):
    # 简单的 OCR 逻辑保持不变
    gx1, gy1, gx2, gy2 = map(int, gauge_box)
    h, w = img.shape[:2]
    pad = 15
    rx1, ry1 = max(0, gx1-pad), max(0, gy1-pad)
    rx2, ry2 = min(w, gx2+pad), min(h, gy2+pad)
    roi = img[ry1:ry2, rx1:rx2]
    
    if roi.size == 0: return None, None

    # 图像增强
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blur)
    
    # EasyOCR 识别
    results = reader.readtext(clahe, detail=1, paragraph=False)
    
    candidates = []
    for (bbox, text, prob) in results:
        val = parse_number(text)
        if val is not None and prob > 0.3:
            cx = (bbox[0][0] + bbox[2][0])/2 + rx1
            cy = (bbox[0][1] + bbox[2][1])/2 + ry1
            candidates.append({'val': val, 'center': (cx, cy)})
    
    if len(candidates) < 2: return None, None 
        
    vmin = min(candidates, key=lambda x: dist_sq(pt_min, x['center']))['val']
    vmax = min(candidates, key=lambda x: dist_sq(pt_max, x['center']))['val']
    
    if vmax < vmin: vmin, vmax = vmax, vmin
    return vmin, vmax

# -----------------------------------------------------------
# 主流程
# -----------------------------------------------------------
def process_gauge(weights, source, output):
    print(f"[Info] 加载模型: {weights}")
    model = YOLO(weights)
    img = cv2.imread(source)
    if img is None: return

    results = model(img, verbose=False)[0]
    boxes = results.boxes.data.cpu().numpy()
    
    gauges = boxes[boxes[:, 5] == CLS_GAUGE]
    print(f"[Info] 检测到 {len(gauges)} 个表盘")
    
    for i, g_box in enumerate(gauges):
        gx1, gy1, gx2, gy2 = g_box[:4]
        
        # 获取关键点 (带简单的范围过滤)
        def get_pt(cls_id):
            c_boxes = boxes[boxes[:, 5] == cls_id]
            valid = [b for b in c_boxes if (gx1-20)<(b[0]+b[2])/2<(gx2+20) and (gy1-20)<(b[1]+b[3])/2<(gy2+20)]
            if not valid: return None
            best = max(valid, key=lambda x: x[4])
            return ((best[0]+best[2])/2, (best[1]+best[3])/2)

        pt_c   = get_pt(CLS_CENTER)
        pt_min = get_pt(CLS_MIN)
        pt_max = get_pt(CLS_MAX)
        pt_tip = get_pt(CLS_TIP)
        
        cv2.rectangle(img, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 255, 0), 2)
        
        if not (pt_c and pt_min and pt_max and pt_tip):
            print(f"  -> Gauge {i+1}: 关键点缺失")
            continue
            
        # 1. OCR 识别量程
        vmin, vmax = get_range_values(img, g_box[:4], pt_min, pt_max)
        if vmin is None:
            vmin, vmax = 0.0, 1.6 # 默认值
            print(f"  -> Gauge {i+1}: OCR 失败，使用默认量程")

        # 2. 计算读数 (使用优化后的函数)
        value = calculate_reading_optimized(pt_c, pt_min, pt_max, pt_tip, vmin, vmax)
        
        print(f"  -> Gauge {i+1}: 读数={value:.3f} (量程 {vmin}~{vmax})")
        
        # 3. 绘图
        # 画出 Min(蓝) Max(红) Center(绿) Tip连线
        cv2.circle(img, (int(pt_min[0]), int(pt_min[1])), 5, (255, 0, 0), -1)
        cv2.circle(img, (int(pt_max[0]), int(pt_max[1])), 5, (0, 0, 255), -1)
        cv2.circle(img, (int(pt_c[0]), int(pt_c[1])), 5, (0, 255, 0), -1)
        cv2.line(img, (int(pt_c[0]), int(pt_c[1])), (int(pt_tip[0]), int(pt_tip[1])), (0, 0, 255), 2)
        
        # 显示数值
        cv2.putText(img, f"{value:.2f}", (int(gx1), int(gy1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(img, f"Range: {vmin}-{vmax}", (int(gx1), int(gy2)+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(output, img)
    print(f"[Done] 保存至: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, default="result_optimized.jpg")
    args = parser.parse_args()
    
    process_gauge(args.weights, args.source, args.output)