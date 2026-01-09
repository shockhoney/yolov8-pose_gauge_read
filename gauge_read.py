import os
import re
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------
# class ids (你的数据集)
# 0 center, 1 gauge, 2 max, 3 min, 4 tip
# ----------------------------
CLS_CENTER = 0
CLS_GAUGE  = 1
CLS_MAX    = 2
CLS_MIN    = 3
CLS_TIP    = 4

# ---------- geometry ----------
def angle_deg(cx, cy, x, y):
    """返回 0~360°，0°向右，90°向上（图像y向下，所以用 -dy）"""
    dx = x - cx
    dy = y - cy
    a = math.degrees(math.atan2(-dy, dx))
    return (a + 360) % 360

def gauge_progress(theta_tip, theta_min, theta_max):
    """
    自动判断顺/逆时针，返回 p∈[0,1] 与方向 'cw'/'ccw'
    通常表盘刻度弧不会超过 270°，用这个规则选方向更稳
    """
    cw_total = (theta_max - theta_min) % 360
    cw_cur   = (theta_tip - theta_min) % 360

    ccw_total = (theta_min - theta_max) % 360
    ccw_cur   = (theta_min - theta_tip) % 360

    if 0 < cw_total <= 270:
        p = cw_cur / cw_total
        direction = "cw"
    else:
        p = (ccw_cur / ccw_total) if ccw_total > 0 else 0.0
        direction = "ccw"

    p = float(np.clip(p, 0.0, 1.0))
    return p, direction

# ---------- OCR helpers ----------
NUM_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)")

def parse_number(text: str):
    text = text.replace(",", ".")
    m = NUM_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except:
        return None

def detect_unit(texts):
    joined = " ".join(texts).lower()
    if "mpa" in joined:
        return "MPa"
    if "kpa" in joined:
        return "kPa"
    if "bar" in joined:
        return "bar"
    if "psi" in joined:
        return "psi"
    return ""

def build_ring_mask(h, w, r_in=0.55, r_out=0.98):
    """只保留表盘外圈（数字常在外圈），减少干扰"""
    cy, cx = h / 2, w / 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    r = min(h, w) / 2
    m = (dist >= r_in * r) & (dist <= r_out * r)
    return (m.astype(np.uint8) * 255)

def openocr_read(img_bgr):
    """
    使用 openocr-python 的 OpenOCR 做 OCR。
    由于不同封装版本返回结构可能不同，这里做了“多形态解析”：
    - list: [ [quad, text, score], ... ]
    - dict: {'results': ...} / {'data': ...} 等
    """
    try:
        from openocr import OpenOCR
    except Exception as e:
        raise RuntimeError("未找到 openocr 模块，请确认已在当前环境安装 openocr-python") from e

    ocr = OpenOCR()

    # 有的版本是 ocr(img)，有的是 ocr.ocr(img)
    try:
        res = ocr(img_bgr)
    except Exception:
        res = ocr.ocr(img_bgr)

    # 统一提取成 items: {text, value, center(x,y), conf}
    items = []

    def handle_list(lst):
        for r in lst:
            # 期望 r ≈ [quad, text, score] 或 (quad, text, score)
            if not isinstance(r, (list, tuple)) or len(r) < 2:
                continue
            quad = r[0]
            text = r[1]
            score = r[2] if len(r) >= 3 else 1.0
            try:
                quad = np.array(quad, dtype=float)
                if quad.ndim == 2 and quad.shape[0] >= 4:
                    cx = float(quad[:, 0].mean())
                    cy = float(quad[:, 1].mean())
                else:
                    # 如果给的是 bbox [x1,y1,x2,y2]
                    quad = np.array(quad, dtype=float).reshape(-1)
                    cx = float((quad[0] + quad[2]) / 2)
                    cy = float((quad[1] + quad[3]) / 2)
                val = parse_number(str(text))
                items.append({"text": str(text), "value": val, "center": (cx, cy), "conf": float(score)})
            except Exception:
                continue

    if isinstance(res, dict):
        # 常见 key
        for k in ["results", "data", "result"]:
            if k in res and isinstance(res[k], list):
                handle_list(res[k])
                break
        else:
            # 如果 dict 结构更怪，直接返回空（让上层兜底）
            return []
    elif isinstance(res, list):
        handle_list(res)
    else:
        return []

    return items

def choose_vmin_vmax(ocr_items, min_pt_roi, max_pt_roi):
    """用“离 MINP/MAXP 最近”匹配量程端点，更稳"""
    nums = [it for it in ocr_items if it["value"] is not None]
    if not nums:
        return 0.0, None

    def d2(p, q):
        return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2

    vmin = None
    vmax = None

    if min_pt_roi is not None:
        vmin = min(nums, key=lambda it: d2(it["center"], min_pt_roi))["value"]
    if max_pt_roi is not None:
        vmax = min(nums, key=lambda it: d2(it["center"], max_pt_roi))["value"]

    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = max(it["value"] for it in nums)

    if vmax < vmin:
        vmin, vmax = vmax, vmin

    return float(vmin), float(vmax)

# ---------- detection helpers ----------
def boxes_to_numpy(r):
    """返回 Nx6: x1,y1,x2,y2,conf,cls"""
    xyxy = r.boxes.xyxy.cpu().numpy()
    cf = r.boxes.conf.cpu().numpy().reshape(-1, 1)
    cl = r.boxes.cls.cpu().numpy().reshape(-1, 1)
    return np.concatenate([xyxy, cf, cl], axis=1)

def center_of_xyxy(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2

def pick_best_in_gauge(all_boxes, gauge_xyxy, cls_id):
    """在某个 gauge bbox 内，选该类最高置信度的框"""
    gx1, gy1, gx2, gy2 = gauge_xyxy
    cand = all_boxes[all_boxes[:, 5] == cls_id]
    if len(cand) == 0:
        return None

    # 只保留中心点落在 gauge 框内的
    keep = []
    for b in cand:
        x1, y1, x2, y2, conf, cls = b
        cx, cy = center_of_xyxy(x1, y1, x2, y2)
        if gx1 <= cx <= gx2 and gy1 <= cy <= gy2:
            keep.append(b)
    if not keep:
        return None

    keep = np.stack(keep, axis=0)
    return keep[np.argmax(keep[:, 4])]

# ---------- visualization ----------
def draw_text_outline(img, text, org, scale=1.2, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def process_one_image(model, img_path, out_path, conf=0.25, pad=10, ring_in=0.55, ring_out=0.98):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] 读不到图片: {img_path}")
        return

    r = model.predict(img, conf=conf, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        print(f"[WARN] 无检测框: {img_path}")
        cv2.imwrite(str(out_path), img)
        return

    boxes = boxes_to_numpy(r)
    gauges = boxes[boxes[:, 5] == CLS_GAUGE]
    if len(gauges) == 0:
        print(f"[WARN] 未检测到 gauge(类1): {img_path}")
        cv2.imwrite(str(out_path), img)
        return

    # 置信度排序，从高到低处理（支持一图多表）
    gauges = gauges[np.argsort(-gauges[:, 4])]

    vis = img.copy()
    results_text = []

    for gi, gb in enumerate(gauges):
        gx1, gy1, gx2, gy2, gconf, _ = gb
        gx1i = max(0, int(gx1 - pad))
        gy1i = max(0, int(gy1 - pad))
        gx2i = min(img.shape[1], int(gx2 + pad))
        gy2i = min(img.shape[0], int(gy2 + pad))

        # 在该 gauge 内找 center/min/max/tip
        b_center = pick_best_in_gauge(boxes, (gx1, gy1, gx2, gy2), CLS_CENTER)
        b_min    = pick_best_in_gauge(boxes, (gx1, gy1, gx2, gy2), CLS_MIN)
        b_max    = pick_best_in_gauge(boxes, (gx1, gy1, gx2, gy2), CLS_MAX)
        b_tip    = pick_best_in_gauge(boxes, (gx1, gy1, gx2, gy2), CLS_TIP)

        if any(b is None for b in [b_center, b_min, b_max, b_tip]):
            # 这个 gauge 信息不全，跳过
            continue

        # 取四点中心坐标（全图坐标）
        cx, cy = center_of_xyxy(*b_center[:4])
        mnx, mny = center_of_xyxy(*b_min[:4])
        mxx, mxy = center_of_xyxy(*b_max[:4])
        tx, ty = center_of_xyxy(*b_tip[:4])

        # --- OCR on ROI ---
        roi = img[gy1i:gy2i, gx1i:gx2i].copy()
        h, w = roi.shape[:2]
        mask = build_ring_mask(h, w, r_in=ring_in, r_out=ring_out)
        roi_masked = cv2.bitwise_and(roi, roi, mask=mask)

        ocr_items = openocr_read(roi_masked)
        unit = detect_unit([it["text"] for it in ocr_items])

        # min/max 点映射到 ROI 坐标
        min_pt_roi = (mnx - gx1i, mny - gy1i)
        max_pt_roi = (mxx - gx1i, mxy - gy1i)

        vmin, vmax = choose_vmin_vmax(ocr_items, min_pt_roi, max_pt_roi)
        if vmax is None:
            continue

        # --- angle → progress p → value ---
        th_tip = angle_deg(cx, cy, tx, ty)
        th_min = angle_deg(cx, cy, mnx, mny)
        th_max = angle_deg(cx, cy, mxx, mxy)
        p, direction = gauge_progress(th_tip, th_min, th_max)
        value = vmin + p * (vmax - vmin)

        # --- draw ---
        cv2.rectangle(vis, (gx1i, gy1i), (gx2i, gy2i), (0, 255, 0), 2)

        # draw points
        def draw_pt(pt, name):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(vis, name, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        draw_pt((cx, cy), "C")
        draw_pt((mnx, mny), "MIN")
        draw_pt((mxx, mxy), "MAX")
        draw_pt((tx, ty), "TIP")

        cv2.line(vis, (int(cx), int(cy)), (int(tx), int(ty)), (0, 0, 255), 3)

        txt = f"{value:.3f} {unit}".strip()
        rng = f"range {vmin:g}~{vmax:g} {unit}".strip()
        info = f"g{gi}: {txt}  p={p:.3f} {direction}"

        # 文本位置放在 gauge 框上方
        draw_text_outline(vis, txt, (gx1i + 10, max(30, gy1i - 10)), scale=1.2, thickness=2)
        draw_text_outline(vis, rng, (gx1i + 10, max(60, gy1i + 30)), scale=0.9, thickness=2)

        results_text.append(info)

    cv2.imwrite(str(out_path), vis)
    if results_text:
        print(f"[OK] {img_path.name}")
        for t in results_text:
            print("   ", t)
        print("   saved ->", out_path)
    else:
        print(f"[WARN] {img_path.name}: gauge 检测到了，但缺少 center/min/max/tip 或 OCR 未识别到量程")
        print("   saved ->", out_path)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="detect模型权重 best.pt")
    ap.add_argument("--source", required=True, help="图片路径 或 图片目录")
    ap.add_argument("--outdir", default="out_read", help="输出目录")
    ap.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    ap.add_argument("--pad", type=int, default=10, help="gauge ROI padding")
    args = ap.parse_args()

    model = YOLO(args.weights)

    src = Path(args.source)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if src.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        imgs = [p for p in src.rglob("*") if p.suffix.lower() in exts]
        imgs.sort()
        print(f"[INFO] found {len(imgs)} images")
        for p in imgs:
            out_path = outdir / (p.stem + "_read.jpg")
            process_one_image(model, p, out_path, conf=args.conf, pad=args.pad)
    else:
        out_path = outdir / (src.stem + "_read.jpg")
        process_one_image(model, src, out_path, conf=args.conf, pad=args.pad)

if __name__ == "__main__":
    main()
