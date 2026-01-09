import cv2
import math
import numpy as np
import re
import argparse
import logging
import sys
from ultralytics import YOLO

# ==========================================
# 0. 环境修复与配置
# ==========================================
# 设置日志级别，屏蔽 PaddleOCR 的调试信息 (替代旧版 show_log=False)
logging.getLogger("ppocr").setLevel(logging.WARNING)

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("请安装 paddleocr: pip install paddleocr")
    sys.exit(1)

# YOLO 类别 ID (根据你的模型设定)
CLS_CENTER = 0
CLS_GAUGE  = 1
CLS_MAX    = 2
CLS_MIN    = 3
CLS_TIP    = 4

# 初始化 OCR
# 修复点：移除了 show_log 参数；保留 use_angle_cls (虽然有警告但兼容性最好)
# 如果想消除警告，可改用 use_textline_orientation=True，但需确认 paddle 版本
try:
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
except Exception as e:
    print(f"OCR 初始化失败，请尝试升级 paddlepaddle: {e}")
    sys.exit(1)

# ==========================================
# 工具函数：数学与几何
# ==========================================
def get_angle(center, point):
    """计算点相对于圆心的角度 (0-360度, X轴正向为0, 顺时针增加)"""
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    angle = math.degrees(math.atan2(dy, dx))
    return (angle + 360) % 360

def dist_sq(p1, p2):
    """计算两点距离的平方"""
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def parse_number(text):
    """正则提取数字 (支持小数、负数)"""
    # 替换常见OCR错误: 逗号转点, 字母l/O转数字
    text = text.replace(',', '.').replace('l', '1').replace('O', '0')
    match = re.search(r"-?\d+(\.\d+)?", text)
    return float(match.group()) if match else None

# ==========================================
# 核心功能 A: 图像处理与 OCR
# ==========================================
def preprocess_roi(roi):
    """步骤 3: OCR 前图像增强"""
    if roi.size == 0: return roi
    # 1. 灰度化
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 2. 高斯去噪
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # 3. CLAHE 局部对比度增强 (应对反光)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    return enhanced

def get_range_values(img, gauge_box, pt_min, pt_max):
    """
    步骤 1-6: 在表盘 ROI 中识别数字，并根据距离匹配 vmin 和 vmax
    """
    gx1, gy1, gx2, gy2 = map(int, gauge_box)
    
    # [步骤 1] 裁剪 ROI
    h, w = img.shape[:2]
    pad = 10
    rx1, ry1 = max(0, gx1-pad), max(0, gy1-pad)
    rx2, ry2 = min(w, gx2+pad), min(h, gy2+pad)
    roi = img[ry1:ry2, rx1:rx2]
    
    if roi.size == 0: return None, None

    # [步骤 3] 预处理
    roi_input = preprocess_roi(roi)
    
    # [步骤 4] PaddleOCR 识别
    # cls=True 用于纠正文本方向
    result = ocr.ocr(roi_input, cls=True)
    
    candidates = []
    if result and result[0]:
        for line in result[0]:
            # line 结构: [ [[x1,y1],...], ("text", score) ]
            box_points = np.array(line[0])
            text = line[1][0]
            
            # [步骤 5] 提取数字
            val = parse_number(text)
            if val is not None:
                # 计算文字框中心 (转换回原图全局坐标)
                cx = np.mean(box_points[:, 0]) + rx1
                cy = np.mean(box_points[:, 1]) + ry1
                candidates.append({'val': val, 'center': (cx, cy)})
    
    if len(candidates) < 2:
        return None, None # 没识别到足够的数字
        
    # [步骤 6] 空间距离匹配
    # 找离 Min 检测点最近的数字
    vmin = min(candidates, key=lambda x: dist_sq(pt_min, x['center']))['val']
    # 找离 Max 检测点最近的数字
    vmax = min(candidates, key=lambda x: dist_sq(pt_max, x['center']))['val']
    
    # 简单校验：通常 vmax > vmin，如果反了则交换
    if vmax < vmin:
        vmin, vmax = vmax, vmin
        
    return vmin, vmax

# ==========================================
# 核心功能 B: 几何计算读数
# ==========================================
def calculate_reading(pt_center, pt_min, pt_max, pt_tip, vmin, vmax):
    """
    步骤 1-4: 计算角度比例并映射数值
    """
    # [步骤 2] 计算三个角度
    ang_min = get_angle(pt_center, pt_min)
    ang_max = get_angle(pt_center, pt_max)
    ang_tip = get_angle(pt_center, pt_tip)
    
    # [步骤 3] 计算比例 p
    # 顺时针总跨度
    cw_total = (ang_max - ang_min + 360) % 360
    # 顺时针当前指针跨度
    cw_curr  = (ang_tip - ang_min + 360) % 360
    
    # 简单判断：如果总跨度合理 (一般仪表盘 < 300度)
    if 10 < cw_total <= 300:
        p = cw_curr / cw_total
    else:
        # 如果 cw_total 极小(接近0)或极大(接近360)，说明 min/max 很接近
        # 可能是圆形表盘刻度正好绕了一圈，或者识别错误
        # 这里做一个简单的兜底：假设满偏
        if cw_total == 0:
            p = 0
        else:
            p = cw_curr / cw_total

    # 钳位：防止指针略微超出 Min/Max 导致读数剧变
    if p > 1.2: p = 0.0 # 视为在 Min 左侧
    elif p > 1.0: p = 1.0 # 视为爆表
    
    # [步骤 4] 线性映射
    value = vmin + p * (vmax - vmin)
    return value

# ==========================================
# 主流程
# ==========================================
def process_gauge(weights_path, source_path, output_path):
    print(f"正在加载模型: {weights_path}")
    model = YOLO(weights_path)
    
    print(f"正在读取图片: {source_path}")
    img = cv2.imread(source_path)
    if img is None: 
        print("图片读取失败")
        return

    # 1. YOLO 检测
    results = model(img, verbose=False)[0]
    boxes = results.boxes.data.cpu().numpy() # [x1,y1,x2,y2,conf,cls]
    
    # 筛选出所有表盘 (Class 1)
    gauges = boxes[boxes[:, 5] == CLS_GAUGE]
    print(f"检测到 {len(gauges)} 个表盘")
    
    for i, g_box in enumerate(gauges):
        gx1, gy1, gx2, gy2 = g_box[:4]
        
        # 辅助函数：在当前 Gauge 框内找关键点
        def get_best_point(cls_id):
            c_boxes = boxes[boxes[:, 5] == cls_id]
            in_gauge = []
            for b in c_boxes:
                cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
                # 稍微放宽一点边界判断
                if (gx1-10) < cx < (gx2+10) and (gy1-10) < cy < (gy2+10):
                    in_gauge.append(b)
            if not in_gauge: return None
            best = max(in_gauge, key=lambda x: x[4])
            return ((best[0]+best[2])/2, (best[1]+best[3])/2)

        # [步骤 B-1] 获取四个关键点
        pt_c   = get_best_point(CLS_CENTER)
        pt_min = get_best_point(CLS_MIN)
        pt_max = get_best_point(CLS_MAX)
        pt_tip = get_best_point(CLS_TIP)
        
        # 绘图基础框
        cv2.rectangle(img, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 255, 0), 2)
        
        if not (pt_c and pt_min and pt_max and pt_tip):
            print(f"Gauge {i}: 关键点缺失，跳过")
            missing = []
            if not pt_c: missing.append("Center")
            if not pt_min: missing.append("Min")
            if not pt_max: missing.append("Max")
            if not pt_tip: missing.append("Tip")
            cv2.putText(img, f"Missing: {','.join(missing)}", (int(gx1), int(gy1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            continue
            
        # [阶段 A] OCR 自动量程
        vmin, vmax = get_range_values(img, g_box[:4], pt_min, pt_max)
        
        origin_source = "OCR"
        if vmin is None:
            print(f"Gauge {i}: OCR 失败，使用默认量程 (0-1.6)")
            vmin, vmax = 0.0, 1.6
            origin_source = "Default"
            
        # [阶段 B] 计算读数
        value = calculate_reading(pt_c, pt_min, pt_max, pt_tip, vmin, vmax)
        
        # [步骤 B-5] 绘图保存
        cv2.line(img, (int(pt_c[0]), int(pt_c[1])), (int(pt_tip[0]), int(pt_tip[1])), (0, 0, 255), 3)
        cv2.circle(img, (int(pt_min[0]), int(pt_min[1])), 4, (255,0,0), -1) # Min点 蓝
        cv2.circle(img, (int(pt_max[0]), int(pt_max[1])), 4, (0,0,255), -1) # Max点 红
        
        info = f"{value:.2f}"
        range_info = f"({vmin}~{vmax})"
        
        print(f"Gauge {i}: 读数={value:.3f}, 量程={range_info}")
        
        cv2.putText(img, info, (int(gx1), int(gy1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, range_info, (int(gx1), int(gy2)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imwrite(output_path, img)
    print(f"处理完成，结果已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="/home/devops/works/analog-gauge-reader/runs/detect/train4/weights/best.pt", required=True, help="YOLO模型路径 (best.pt)")
    parser.add_argument("--source", type=str, default="test.jpg", required=True, help="输入图片路径")
    parser.add_argument("--output", type=str, default="result_view.jpg", help="输出图片路径")
    
    args = parser.parse_args()
    
    process_gauge(args.weights, args.source, args.output)