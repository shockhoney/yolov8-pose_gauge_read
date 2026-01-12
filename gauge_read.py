import cv2
import math
import numpy as np
import re
import argparse
import sys
from ultralytics import YOLO

# -----------------------------------------------------------
# 0. 环境检查与导入
# -----------------------------------------------------------
try:
    import easyocr
except ImportError:
    print("[Error] 缺少 easyocr 库。")
    print("请运行: pip install easyocr")
    sys.exit(1)

# YOLO 类别 ID (根据你的模型设定)
CLS_CENTER = 0
CLS_GAUGE  = 1
CLS_MAX    = 2
CLS_MIN    = 3
CLS_TIP    = 4

# 初始化 OCR Reader
# gpu=True 会自动利用你跑 YOLO 的那个 GPU
print("[Init] 初始化 EasyOCR (PyTorch后端)...")
reader = easyocr.Reader(['en'], gpu=True) 

# -----------------------------------------------------------
# 工具函数
# -----------------------------------------------------------
def get_angle(center, point):
    """计算点相对于圆心的角度 (0-360度, X轴正向为0, 顺时针增加)"""
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    # atan2 返回范围 [-180, 180]
    angle = math.degrees(math.atan2(dy, dx))
    return (angle + 360) % 360

def dist_sq(p1, p2):
    """计算距离平方"""
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def parse_number(text):
    """从文本提取数字，处理常见 OCR 误读"""
    # EasyOCR 常见错误: 'l'->1, 'O'->0, ','->. 
    # 还可以过滤掉非 ASCII 字符
    text = text.replace(',', '.').replace('l', '1').replace('O', '0').lower()
    match = re.search(r"-?\d+(\.\d+)?", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None

# -----------------------------------------------------------
# 核心逻辑 A: OCR 量程识别
# -----------------------------------------------------------
def preprocess_roi(roi):
    """图像增强：让数字在金属反光表面更清晰"""
    if roi.size == 0: return roi
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # CLAHE 局部对比度增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(blur)

def get_range_values(img, gauge_box, pt_min, pt_max):
    """
    裁剪表盘 ROI -> 识别所有数字 -> 根据距离匹配 Min/Max 值
    """
    gx1, gy1, gx2, gy2 = map(int, gauge_box)
    h, w = img.shape[:2]
    
    # 扩大 ROI 防止数字贴边
    pad = 15
    rx1, ry1 = max(0, gx1-pad), max(0, gy1-pad)
    rx2, ry2 = min(w, gx2+pad), min(h, gy2+pad)
    roi = img[ry1:ry2, rx1:rx2]
    if roi.size == 0: return 0.0, 1.6 # 默认值

    # 预处理
    roi_input = preprocess_roi(roi)
    
    # --- EasyOCR 推理 ---
    # detail=1 返回 [bbox, text, conf]
    # paragraph=False 逐行识别
    results = reader.readtext(roi_input, detail=1, paragraph=False)
    
    candidates = []
    for (bbox, text, prob) in results:
        val = parse_number(text)
        
        # 过滤低置信度 (可选)
        if val is not None and prob > 0.2:
            # 转换回原图全局坐标
            cx_global = cx_roi + rx1
            cy_global = cy_roi + ry1
            candidates.append({'val': val, 'center': (cx_global, cy_global)})
    
    if len(candidates) < 2:
        return None, None 
        
    # 空间匹配：谁离 Min 点近，谁就是 vmin
    vmin = min(candidates, key=lambda x: dist_sq(pt_min, x['center']))['val']
    vmax = min(candidates, key=lambda x: dist_sq(pt_max, x['center']))['val']
    
    # 逻辑修正: 通常 vmax > vmin
    if vmax < vmin:
        vmin, vmax = vmax, vmin
        
    return vmin, vmax

# -----------------------------------------------------------
# 核心逻辑 B: 角度计算与读数
# -----------------------------------------------------------
def calculate_reading(pt_center, pt_min, pt_max, pt_tip, vmin, vmax):
    # 1. 计算角度
    ang_min = get_angle(pt_center, pt_min)
    ang_max = get_angle(pt_center, pt_max)
    ang_tip = get_angle(pt_center, pt_tip)
    
    # 2. 计算顺时针跨度
    cw_total = (ang_max - ang_min + 360) % 360
    cw_curr  = (ang_tip - ang_min + 360) % 360
    
    # 3. 计算进度 p
    # 正常表盘刻度范围一般在 90~320 度之间
    if 10 < cw_total <= 340:
        p = cw_curr / cw_total
    else:
        p = 0.0

    # 4. 钳位 (Clamping)
    if p > 1.2: p = 0.0
    elif p > 1.0: p = 1.0
    
    # 5. 线性插值
    return vmin + p * (vmax - vmin)

# -----------------------------------------------------------
# 主程序
# -----------------------------------------------------------
def process_gauge(weights, source, output):
    print(f"[Info] 加载 YOLO 模型: {weights}")
    try:
        model = YOLO(weights)
    except Exception as e:
        print(f"[Error] YOLO 加载失败: {e}")
        return

    print(f"[Info] 读取图片: {source}")
    img = cv2.imread(source)
    if img is None: return

    results = model(img, verbose=False)[0]
    boxes = results.boxes.data.cpu().numpy()
    
    gauges = boxes[boxes[:, 5] == CLS_GAUGE]
    print(f"[Info] 检测到 {len(gauges)} 个仪表盘")
    
    for i, g_box in enumerate(gauges):
        gx1, gy1, gx2, gy2 = g_box[:4]
        
        # 寻找关键点 (在 gauge 框内)
        def get_pt(cls_id):
            c_boxes = boxes[boxes[:, 5] == cls_id]
            valid = []
            for b in c_boxes:
                cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
                if (gx1-20) < cx < (gx2+20) and (gy1-20) < cy < (gy2+20):
                    valid.append(b)
            if not valid: return None
            # 取置信度最高的
            best = max(valid, key=lambda x: x[4])
            return ((best[0]+best[2])/2, (best[1]+best[3])/2)

        pt_c   = get_pt(CLS_CENTER)
        pt_min = get_pt(CLS_MIN)
        pt_max = get_pt(CLS_MAX)
        pt_tip = get_pt(CLS_TIP)
        
        # 绘图
        cv2.rectangle(img, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 255, 0), 2)

        if not (pt_c and pt_min and pt_max and pt_tip):
            print(f"  -> Gauge {i+1}: 关键点缺失，跳过")
            continue
            
        # --- 阶段 A: EasyOCR ---
        vmin, vmax = get_range_values(img, g_box[:4], pt_min, pt_max)
        
        if vmin is None:
            print(f"  -> Gauge {i+1}: OCR 未识别到量程，使用默认 (0-1.6)")
            vmin, vmax = 0.0, 1.6
            
        # --- 阶段 B: 读数 ---
        value = calculate_reading(pt_c, pt_min, pt_max, pt_tip, vmin, vmax)
        
        print(f"  -> Gauge {i+1}: 读数={value:.3f} (量程 {vmin}~{vmax})")
        
        # 可视化
        cv2.line(img, (int(pt_c[0]), int(pt_c[1])), (int(pt_tip[0]), int(pt_tip[1])), (0, 0, 255), 2)
        # 标记 Min/Max 匹配点
        cv2.circle(img, (int(pt_min[0]), int(pt_min[1])), 5, (255,0,0), -1)
        cv2.circle(img, (int(pt_max[0]), int(pt_max[1])), 5, (0,0,255), -1)
        
        cv2.putText(img, f"{value:.2f}", (int(gx1), int(gy1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"R: {vmin}-{vmax}", (int(gx1), int(gy2)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    cv2.imwrite(output, img)
    print(f"[Done] 结果已保存: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="/home/devops/works/analog-gauge-reader/runs/detect/train4/weights/best.pt", type=str, required=True)
    parser.add_argument("--source", default="test.jpg", type=str, required=True)
    parser.add_argument("--output", type=str, default="result_view.jpg")
    args = parser.parse_args()
    
    process_gauge(args.weights, args.source, args.output)
