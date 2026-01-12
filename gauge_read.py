import cv2
import math
import numpy as np
import re
import argparse
import sys
from ultralytics import YOLO

# -----------------------------------------------------------
# 环境初始化
# -----------------------------------------------------------
try:
    import easyocr
except ImportError:
    print("[Error] 请先安装 easyocr: pip install easyocr")
    sys.exit(1)

print("[Init] 初始化 OCR 引擎...")
reader = easyocr.Reader(['en'], gpu=True)

CLS_CENTER, CLS_GAUGE, CLS_MAX, CLS_MIN, CLS_TIP = 0, 1, 2, 3, 4

# -----------------------------------------------------------
# 核心逻辑：智能角度计算
# -----------------------------------------------------------
def get_angle(center, point):
    """计算绝对角度 (0-360)，X轴正向为0，顺时针增加"""
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    angle = math.degrees(math.atan2(dy, dx))
    return (angle + 360) % 360

def calculate_value_smart(pt_c, pt_min, pt_max, pt_tip, vmin, vmax):
    """
    【智能读数算法】
    1. 计算 Min/Max 原始角度
    2. 判断是否符合标准 270度 表盘特征
    3. 如果符合，使用虚拟刻度修正 Min/Max 角度，消除文字框偏移带来的误差
    """
    ang_min_raw = get_angle(pt_c, pt_min)
    ang_max_raw = get_angle(pt_c, pt_max)
    ang_tip     = get_angle(pt_c, pt_tip)

    # 1. 计算原始跨度 (顺时针 Min -> Max)
    span_raw = (ang_max_raw - ang_min_raw + 360) % 360

    # 2. 计算中轴线角度 (指向表盘底部空白处)
    #    它是 Min 和 Max 的角平分线
    bisector_angle = (ang_min_raw + span_raw / 2) % 360

    # 3. 智能修正逻辑
    #    绝大多数工业表盘是 270度 (3/4圆)
    #    如果原始检测的跨度在 240~300 之间，我们强制将其校准为 270度
    #    这样可以修复 "数字框中心" 与 "刻度线" 不重合导致的误差
    STANDARD_SPAN = 270.0
    
    if 240 < span_raw < 300:
        # 使用虚拟刻度
        # print(f"  [Debug] 启用 270度 标准表盘修正 (原始跨度: {span_raw:.1f}°)")
        virtual_start = (bisector_angle - STANDARD_SPAN / 2 + 360) % 360
        # virtual_end   = (bisector_angle + STANDARD_SPAN / 2 + 360) % 360
        
        # 修正后的总量程
        span_total = STANDARD_SPAN
        # 修正后的指针跨度 (相对于虚拟起点)
        span_tip = (ang_tip - virtual_start + 360) % 360
    else:
        # 非标表盘，使用原始检测值
        span_total = span_raw
        span_tip = (ang_tip - ang_min_raw + 360) % 360

    # 4. 读数映射与死区处理
    #    正常范围
    if span_tip <= span_total:
        progress = span_tip / span_total
        return vmin + progress * (vmax - vmin)
    
    #    死区处理 (指针在 Max 和 Min 之间的空白区)
    else:
        dist_to_max = span_tip - span_total
        dist_to_min = 360 - span_tip
        return vmin if dist_to_min < dist_to_max else vmax

# -----------------------------------------------------------
# 辅助逻辑：OCR 与 坐标获取
# -----------------------------------------------------------
def parse_num(text):
    text = text.replace(',', '.').replace('l', '1').replace('O', '0').lower()
    m = re.search(r"-?\d+(\.\d+)?", text)
    return float(m.group()) if m else None

def get_ocr_range(img, bbox, pt_min, pt_max):
    gx1, gy1, gx2, gy2 = map(int, bbox)
    h, w = img.shape[:2]
    pad = 10
    roi = img[max(0,gy1-pad):min(h,gy2+pad), max(0,gx1-pad):min(w,gx2+pad)]
    
    if roi.size == 0: return 0.0, 1.6 # 默认兜底

    # 简单增强
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    
    res = reader.readtext(enhanced, detail=1)
    cands = []
    for (box, txt, _) in res:
        v = parse_num(txt)
        if v is not None:
            cx = (box[0][0]+box[2][0])/2 + max(0,gx1-pad)
            cy = (box[0][1]+box[2][1])/2 + max(0,gy1-pad)
            cands.append({'v':v, 'c':(cx,cy)})
            
    if len(cands) < 2: return 0.0, 1.6
    
    d2 = lambda p1,p2: (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    vmin = min(cands, key=lambda x: d2(pt_min, x['c']))['v']
    vmax = min(cands, key=lambda x: d2(pt_max, x['c']))['v']
    
    if vmax < vmin: vmin, vmax = vmax, vmin
    return vmin, vmax

def process_gauge(weights, source, output):
    model = YOLO(weights)
    img = cv2.imread(source)
    if img is None: return

    res = model(img, verbose=False)[0]
    boxes = res.boxes.data.cpu().numpy()
    
    gauges = boxes[boxes[:, 5] == CLS_GAUGE]
    print(f"[Info] 检测到 {len(gauges)} 个仪表")

    for i, g_box in enumerate(gauges):
        gx1, gy1, gx2, gy2 = g_box[:4]
        
        # 获取关键点 (在 gauge 框内)
        def get_pt(cid):
            cbs = [b for b in boxes[boxes[:, 5]==cid] 
                   if (gx1-20)<(b[0]+b[2])/2<(gx2+20) and (gy1-20)<(b[1]+b[3])/2<(gy2+20)]
            if not cbs: return None
            best = max(cbs, key=lambda x: x[4])
            return ((best[0]+best[2])/2, (best[1]+best[3])/2)

        pt_c, pt_min, pt_max, pt_tip = [get_pt(i) for i in [CLS_CENTER, CLS_MIN, CLS_MAX, CLS_TIP]]

        # 绘图基础
        cv2.rectangle(img, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 255, 0), 2)
        
        if not all([pt_c, pt_min, pt_max, pt_tip]):
            print(f"  -> Gauge {i+1}: 关键点不全，跳过")
            continue

        # 1. OCR 量程
        vmin, vmax = get_ocr_range(img, g_box[:4], pt_min, pt_max)
        
        # 2. 智能读数
        value = calculate_value_smart(pt_c, pt_min, pt_max, pt_tip, vmin, vmax)
        print(f"  -> Gauge {i+1}: 读数={value:.3f} (量程 {vmin}~{vmax})")

        # 3. 结果上图
        # 指针线
        cv2.line(img, (int(pt_c[0]), int(pt_c[1])), (int(pt_tip[0]), int(pt_tip[1])), (0, 0, 255), 3)
        # 数值显示
        text = f"{value:.2f}"
        cv2.putText(img, text, (int(gx1), int(gy1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(img, f"Range: {vmin}-{vmax}", (int(gx1), int(gy2)+25), 0, 0.7, (200,200,200), 2)

    cv2.imwrite(output, img)
    print(f"[Done] 保存至: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, default="result_optimized.jpg")
    args = parser.parse_args()
    
    process_gauge(args.weights, args.source, args.output)