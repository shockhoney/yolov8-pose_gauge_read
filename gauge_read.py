import cv2
import math
import numpy as np
import re
import argparse
from ultralytics import YOLO
from paddleocr import PaddleOCR


CLS_CENTER = 0
CLS_GAUGE  = 1
CLS_MAX    = 2
CLS_MIN    = 3
CLS_TIP    = 4


ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)


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
    text = text.replace(',', '.') # 修正OCR常见的逗号误读
    match = re.search(r"-?\d+(\.\d+)?", text)
    return float(match.group()) if match else None


def preprocess_roi(roi):
    """步骤 3: OCR 前图像增强"""
    # 1. 灰度化
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 2. 高斯去噪
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # 3. CLAHE 局部对比度增强 (应对反光)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    # 4. 二值化 (可选，PaddleOCR自带二值化，但增强对比度很有用)
    return enhanced

def get_range_values(img, gauge_box, pt_min, pt_max):

    gx1, gy1, gx2, gy2 = map(int, gauge_box)
    
    # [步骤 1] 裁剪 ROI
    h, w = img.shape[:2]
    # 稍微扩大一点 ROI 防止数字被切掉
    pad = 10
    rx1, ry1 = max(0, gx1-pad), max(0, gy1-pad)
    rx2, ry2 = min(w, gx2+pad), min(h, gy2+pad)
    roi = img[ry1:ry2, rx1:rx2]
    
    # [步骤 2] 环形 Mask (可选，这里简化为直接识别，依靠距离筛选)
    # 如果背景干扰极大，可在此处 mask 掉 ROI 中心区域
    
    # [步骤 3] 预处理
    roi_input = preprocess_roi(roi)
    
    # [步骤 4] OpenOCR 识别
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
    
    # 简单校验：通常 vmax > vmin
    if vmax < vmin:
        vmin, vmax = vmax, vmin
        
    return vmin, vmax

def calculate_reading(pt_center, pt_min, pt_max, pt_tip, vmin, vmax):
    """
    步骤 1-4: 计算角度比例并映射数值
    """
    # [步骤 2] 计算三个角度
    ang_min = get_angle(pt_center, pt_min)
    ang_max = get_angle(pt_center, pt_max)
    ang_tip = get_angle(pt_center, pt_tip)
    
    # [步骤 3] 计算比例 p (处理 0/360 度跨越)
    # 顺时针总跨度
    cw_total = (ang_max - ang_min + 360) % 360
    # 顺时针当前指针跨度
    cw_curr  = (ang_tip - ang_min + 360) % 360
    
    # 简单判断：如果总跨度合理 (一般仪表盘 < 270度)
    if 0 < cw_total <= 300:
        p = cw_curr / cw_total
    else:
        # 可能是逆时针表盘或检测出错，这里默认顺时针处理
        p = 0
    
    # 钳位：防止指针略微超出 Min/Max 导致读数剧变
    # 如果 p > 1.2，说明指针可能在 Min 左边 (也就是小于 vmin)
    if p > 1.2: p = 0.0
    if p > 1.0: p = 1.0
    
    # [步骤 4] 线性映射
    value = vmin + p * (vmax - vmin)
    return value

def process_gauge(model_path, img_path):
    # 加载模型
    model = YOLO(model_path)
    img = cv2.imread(img_path)
    if img is None: return

    # 1. YOLO 检测
    results = model(img)[0]
    boxes = results.boxes.data.cpu().numpy() # [x1,y1,x2,y2,conf,cls]
    
    # 筛选出所有表盘 (Class 1)
    gauges = boxes[boxes[:, 5] == CLS_GAUGE]
    
    for g_box in gauges:
        gx1, gy1, gx2, gy2 = g_box[:4]
        
        # 辅助函数：在当前 Gauge 框内找关键点
        def get_best_point(cls_id):
            # 找到所有该类别的框
            c_boxes = boxes[boxes[:, 5] == cls_id]
            # 筛选中心点在 Gauge 内部的
            in_gauge = []
            for b in c_boxes:
                cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
                if gx1 < cx < gx2 and gy1 < cy < gy2:
                    in_gauge.append(b)
            # 返回置信度最高的中心点
            if not in_gauge: return None
            best = max(in_gauge, key=lambda x: x[4])
            return ((best[0]+best[2])/2, (best[1]+best[3])/2)

        # [步骤 B-1] 获取四个关键点
        pt_c   = get_best_point(CLS_CENTER)
        pt_min = get_best_point(CLS_MIN)
        pt_max = get_best_point(CLS_MAX)
        pt_tip = get_best_point(CLS_TIP)
        
        # 如果关键点缺失，无法计算
        if not (pt_c and pt_min and pt_max and pt_tip):
            print("关键点缺失，跳过此表盘")
            continue
            
        # [阶段 A] OCR 自动量程
        vmin, vmax = get_range_values(img, g_box[:4], pt_min, pt_max)
        
        if vmin is None:
            print("OCR 未识别到足够数字，使用默认量程 0-1.6")
            vmin, vmax = 0.0, 1.6
            
        # [阶段 B] 计算读数
        value = calculate_reading(pt_c, pt_min, pt_max, pt_tip, vmin, vmax)
        
        # [步骤 B-5] 绘图保存
        # 画框
        cv2.rectangle(img, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 255, 0), 2)
        # 画指针
        cv2.line(img, (int(pt_c[0]), int(pt_c[1])), (int(pt_tip[0]), int(pt_tip[1])), (0, 0, 255), 3)
        # 写文字
        info = f"Val: {value:.2f} ({vmin}-{vmax})"
        print(f"读数结果: {info}")
        cv2.putText(img, info, (int(gx1), int(gy1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imwrite("result_final.jpg", img)
    print("结果已保存至 result_final.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="/home/devops/works/analog-gauge-reader/runs/detect/train4/weights/best.pt", help="YOLO模型路径")
    parser.add_argument("--source", type=str, default="test.jpg", help="图片路径")
    args = parser.parse_args()
    
    process_gauge(args.weights, args.source)