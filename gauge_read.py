import cv2
import math
import numpy as np
import re
import argparse
import sys
from ultralytics import YOLO
import easyocr

# -----------------------------------------------------------
# 0. 环境初始化
# -----------------------------------------------------------
print("[Init] 初始化 OCR...")
reader = easyocr.Reader(['en'], gpu=True) 

CLS_CENTER, CLS_GAUGE, CLS_MAX, CLS_MIN, CLS_TIP = 0, 1, 2, 3, 4

# -----------------------------------------------------------
# 1. 基础几何计算
# -----------------------------------------------------------
def get_angle(center, point):
    """计算绝对角度 0-360，X轴正向=0，顺时针增加"""
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    angle = math.degrees(math.atan2(dy, dx))
    return (angle + 360) % 360

def calculate_value_strict(pt_c, pt_min, pt_max, pt_tip, vmin, vmax):
    """
    【严格几何读数】
    完全基于 YOLO 检测到的 Min/Max/Tip 三点物理位置进行计算。
    不进行任何人为的几何校准。
    """
    # 1. 获取三个点的绝对角度
    ang_min = get_angle(pt_c, pt_min)
    ang_max = get_angle(pt_c, pt_max)
    ang_tip = get_angle(pt_c, pt_tip)

    # 2. 计算量程总跨度 (顺时针 Min -> Max)
    total_span = (ang_max - ang_min + 360) % 360
    
    # 3. 计算指针当前跨度 (顺时针 Min -> Tip)
    tip_span = (ang_tip - ang_min + 360) % 360

    # 4. 异常保护 (跨度太小认为无效)
    if total_span < 10:
        return vmin

    # 5. 读数计算与死区处理
    if tip_span <= total_span:
        # 正常情况：指针在 Min 和 Max 夹角之间
        progress = tip_span / total_span
        return vmin + progress * (vmax - vmin)
    else:
        # 死区情况：指针在 Max 和 Min 之间的空白区域
        dist_to_min = 360 - tip_span   # 继续转一圈回到 Min 的距离
        dist_to_max = tip_span - total_span # 刚过 Max 的距离
        
        if dist_to_min < dist_to_max:
            return vmin
        else:
            return vmax

# -----------------------------------------------------------
# 2. OCR 与 辅助功能
# -----------------------------------------------------------
def parse_num(txt):
    """清洗 OCR 文本"""
    txt = txt.replace(',','.').replace('l','1').replace('O','0').lower()
    m = re.search(r"-?\d+(\.\d+)?", txt)
    return float(m.group()) if m else None

def get_ocr_range(img, bbox, pt_min, pt_max):
    """识别并匹配量程"""
    gx1, gy1, gx2, gy2 = map(int, bbox)
    h, w = img.shape[:2]
    pad = 15
    roi = img[max(0,gy1-pad):min(h,gy2+pad), max(0,gx1-pad):min(w,gx2+pad)]
    if roi.size == 0: return 0.0, 1.6

    # 图像增强
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    
    # OCR 识别
    res = reader.readtext(enhanced, detail=1, paragraph=False)
    cands = []
    for (box, txt, prob) in res:
        v = parse_num(txt)
        if v is not None and prob > 0.3:
            # 将 ROI 坐标转回全局坐标
            cx = (box[0][0]+box[2][0])/2 + max(0,gx1-pad)
            cy = (box[0][1]+box[2][1])/2 + max(0,gy1-pad)
            cands.append({'v':v, 'c':(cx,cy)})
            
    if len(cands) < 2: return 0.0, 1.6
    
    # 空间距离匹配 (离物理点最近的数字)
    d2 = lambda p1,p2: (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    vmin = min(cands, key=lambda x: d2(pt_min, x['c']))['v']
    vmax = min(cands, key=lambda x: d2(pt_max, x['c']))['v']
    
    if vmax < vmin: vmin, vmax = vmax, vmin
    return vmin, vmax

# -----------------------------------------------------------
# 3. 主程序
# -----------------------------------------------------------
def process_gauge(weights, source, output):
    print(f"[Info] 加载 YOLO: {weights}")
    model = YOLO(weights)
    img = cv2.imread(source)
    if img is None: return

    res = model(img, verbose=False)[0]
    boxes = res.boxes.data.cpu().numpy()
    gauges = boxes[boxes[:, 5] == CLS_GAUGE]
    print(f"[Info] 检测到 {len(gauges)} 个表盘")

    for i, g_box in enumerate(gauges):
        gx1, gy1, gx2, gy2 = g_box[:4]
        
        # 筛选 Gauge 内部的关键点
        def get_pt(cid):
            cbs = [b for b in boxes[boxes[:, 5]==cid] 
                   if (gx1-20)<(b[0]+b[2])/2<(gx2+20) and (gy1-20)<(b[1]+b[3])/2<(gy2+20)]
            if not cbs: return None
            best = max(cbs, key=lambda x: x[4])
            return ((best[0]+best[2])/2, (best[1]+best[3])/2)

        pt_c   = get_pt(CLS_CENTER)
        pt_min = get_pt(CLS_MIN)
        pt_max = get_pt(CLS_MAX)
        pt_tip = get_pt(CLS_TIP)
        
        # 画表盘框
        cv2.rectangle(img, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 255, 0), 2)
        
        if not all([pt_c, pt_min, pt_max, pt_tip]):
            print(f"Skipping Gauge {i}: 关键点不全")
            continue

        # 1. OCR 获取量程
        vmin, vmax = get_ocr_range(img, g_box[:4], pt_min, pt_max)
        
        # 2. 计算读数
        value = calculate_value_strict(pt_c, pt_min, pt_max, pt_tip, vmin, vmax)
        print(f"Gauge {i}: Val={value:.3f} (Range {vmin}-{vmax})")
        
        # --- 可视化 (统一定义颜色：Min蓝, Max红, Tip绿) ---
        
        # 画线
        cv2.line(img, (int(pt_c[0]), int(pt_c[1])), (int(pt_min[0]), int(pt_min[1])), (255, 0, 0), 2) # Min线: 蓝
        cv2.line(img, (int(pt_c[0]), int(pt_c[1])), (int(pt_max[0]), int(pt_max[1])), (0, 0, 255), 2) # Max线: 红
        cv2.line(img, (int(pt_c[0]), int(pt_c[1])), (int(pt_tip[0]), int(pt_tip[1])), (0, 255, 0), 3) # 指针线: 绿
        
        # 画点
        cv2.circle(img, (int(pt_min[0]), int(pt_min[1])), 5, (255,0,0), -1) # 蓝点
        cv2.circle(img, (int(pt_max[0]), int(pt_max[1])), 5, (0,0,255), -1) # 红点
        cv2.circle(img, (int(pt_tip[0]), int(pt_tip[1])), 5, (0,255,0), -1) # 绿点
        
        # 写字 (调整了大小以适应普通图片)
        # 读数
        cv2.putText(img, f"{value:.3f}", (int(gx1), int(gy1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        # 量程
        cv2.putText(img, f"Range: {vmin}-{vmax}", (int(gx1), int(gy2)+30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imwrite(output, img)
    print(f"Saved: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, default="result_strict.jpg")
    args = parser.parse_args()
    process_gauge(args.weights, args.source, args.output)
