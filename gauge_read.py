import cv2
import math
import numpy as np
import re
import argparse
import sys
from ultralytics import YOLO

# -----------------------------------------------------------
# 环境检查
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
# 核心逻辑：基于中轴线的强制校准算法
# -----------------------------------------------------------
def calculate_angle(center, point):
    """计算绝对角度 (0-360)，X轴正向为0，顺时针增加 (图像坐标系 Y向下)"""
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    angle = math.degrees(math.atan2(dy, dx))
    return (angle + 360) % 360

def get_reading_calibrated(pt_c, pt_min, pt_max, pt_tip, vmin, vmax):
    """
    【强制校准版】
    不直接使用 Min/Max 的角度差作为量程，而是：
    1. 计算 Min 和 Max 的角平分线（表盘中轴线）。
    2. 强制假设量程为 270度 (工业标准)。
    3. 推算出虚拟的、精确的 0刻度 和 满刻度 位置。
    """
    # 1. 获取原始角度
    ang_min = calculate_angle(pt_c, pt_min)
    ang_max = calculate_angle(pt_c, pt_max)
    ang_tip = calculate_angle(pt_c, pt_tip)

    # 2. 计算表盘中轴线 (Symmetry Axis)
    #    方法：利用向量相加找到 Min 和 Max 的中间方向
    vec_min = np.array([pt_min[0]-pt_c[0], pt_min[1]-pt_c[1]])
    vec_max = np.array([pt_max[0]-pt_c[0], pt_max[1]-pt_c[1]])
    # 归一化向量，防止距离影响角度计算
    vec_min = vec_min / np.linalg.norm(vec_min)
    vec_max = vec_max / np.linalg.norm(vec_max)
    
    vec_mid = vec_min + vec_max
    # 计算中轴线角度 (指向表盘底部空缺处)
    ang_mid = calculate_angle((0,0), (vec_mid[0], vec_mid[1]))

    # 3. 强制使用 270度 量程模型
    #    大多数压力表有效刻度是 270度。
    #    Min (0值) 应该在中轴线 逆时针 135度 的位置
    #    Max (满值) 应该在中轴线 顺时针 135度 的位置
    FIXED_SPAN = 270.0
    HALF_SPAN = FIXED_SPAN / 2.0
    
    # 虚拟的精确起止角度
    virtual_start_ang = (ang_mid - HALF_SPAN + 360) % 360
    # virtual_end_ang   = (ang_mid + HALF_SPAN + 360) % 360
    
    # 4. 计算指针相对于虚拟起点的角度
    #    (Tip - Start + 360) % 360
    span_tip = (ang_tip - virtual_start_ang + 360) % 360
    
    # 5. 计算读数
    #    如果指针在 270度 范围内
    if span_tip <= FIXED_SPAN:
        progress = span_tip / FIXED_SPAN
        val = vmin + progress * (vmax - vmin)
        return val, virtual_start_ang # 返回角度用于debug绘图
    else:
        # 死区处理 (指针在底部空缺区)
        dist_to_start = 360 - span_tip
        dist_to_end = span_tip - FIXED_SPAN
        if dist_to_start < dist_to_end:
            return vmin, virtual_start_ang
        else:
            return vmax, virtual_start_ang

# -----------------------------------------------------------
# 辅助：OCR 与 可视化
# -----------------------------------------------------------
def parse_num(txt):
    txt = txt.replace(',','.').replace('l','1').replace('O','0')
    m = re.search(r"-?\d+(\.\d+)?", txt)
    return float(m.group()) if m else None

def get_ocr_range(img, bbox, pt_min, pt_max):
    gx1, gy1, gx2, gy2 = map(int, bbox)
    h, w = img.shape[:2]
    pad = 10
    roi = img[max(0,gy1-pad):min(h,gy2+pad), max(0,gx1-pad):min(w,gx2+pad)]
    if roi.size == 0: return 0.0, 1.6
    
    # OCR 增强
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    
    res = reader.readtext(clahe, detail=1)
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

    for i, g_box in enumerate(gauges):
        gx1, gy1, gx2, gy2 = g_box[:4]
        
        # 获取关键点
        def get_pt(cid):
            cbs = [b for b in boxes[boxes[:, 5]==cid] 
                   if (gx1-20)<(b[0]+b[2])/2<(gx2+20) and (gy1-20)<(b[1]+b[3])/2<(gy2+20)]
            if not cbs: return None
            best = max(cbs, key=lambda x: x[4])
            return ((best[0]+best[2])/2, (best[1]+best[3])/2)

        pt_c, pt_min, pt_max, pt_tip = [get_pt(i) for i in [CLS_CENTER, CLS_MIN, CLS_MAX, CLS_TIP]]
        
        cv2.rectangle(img, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 255, 0), 2)
        if not all([pt_c, pt_min, pt_max, pt_tip]):
            print(f"Skipping Gauge {i}: Missing points")
            continue

        vmin, vmax = get_ocr_range(img, g_box[:4], pt_min, pt_max)
        
        # 计算读数 (带校准)
        value, virtual_start_angle = get_reading_calibrated(pt_c, pt_min, pt_max, pt_tip, vmin, vmax)
        print(f"Gauge {i}: Val={value:.3f} (Range {vmin}-{vmax})")
        
        # 绘图调试
        # 1. 画指针
        cv2.line(img, (int(pt_c[0]), int(pt_c[1])), (int(pt_tip[0]), int(pt_tip[1])), (0, 0, 255), 3)
        # 2. 画计算出的 0刻度起始线 (紫色) - 用于验证逻辑是否对齐了刻度
        r = 30 # 半径
        sx = int(pt_c[0] + r * math.cos(math.radians(virtual_start_angle)))
        sy = int(pt_c[1] + r * math.sin(math.radians(virtual_start_angle)))
        cv2.line(img, (int(pt_c[0]), int(pt_c[1])), (sx, sy), (255, 0, 255), 2)

        # 文本
        cv2.putText(img, f"{value:.2f}", (int(gx1), int(gy1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(img, f"{vmin}-{vmax}", (int(gx1), int(gy2)+25), 0, 0.7, (255, 255, 255), 1)

    cv2.imwrite(output, img)
    print(f"Saved: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, default="result_calibrated.jpg")
    args = parser.parse_args()
    process_gauge(args.weights, args.source, args.output)