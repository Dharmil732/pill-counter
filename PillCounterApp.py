#!/usr/bin/env python3
# PillCounterApp.py — Mobile-ready (phone browser), AI + Classical pill counting with:
# - YOLOv8-seg model mode (optional, free)
# - Classical fallback (Watershed/Contour/Hough)
# - Double-check mode (AI vs classical) with tolerance
# - Scatter-risk detector (tiny fragments %)
# - Confidence heatmap (AI)
# - Barcode tie-in (auto-read if available)
# - Google Sheets sync (via Streamlit Secrets)
# - Mobile Mode defaults (bigger touch targets, safer defaults)
#
# Runs great on Streamlit Cloud (free). If YOLO weights missing, app falls back gracefully.

import os, io, math, base64, json, tempfile, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import streamlit as st

from skimage import morphology, measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# ---------- Optional libraries (we degrade gracefully if missing) ----------
YOLO_AVAILABLE = True
try:
    from ultralytics import YOLO
except Exception:
    YOLO_AVAILABLE = False

BARCODE_AVAILABLE = True
try:
    from pyzbar.pyzbar import decode as zbar_decode
except Exception:
    BARCODE_AVAILABLE = False

SHEETS_AVAILABLE = True
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
except Exception:
    SHEETS_AVAILABLE = False

APP_NAME = "💊 Pill Counting — All-in-One (Mobile)"
OUT_DIR = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(exist_ok=True)
DEFAULT_MODEL_PATH = MODELS_DIR / "pills_yolov8s-seg.pt"  # optional custom weights

st.set_page_config(page_title="Pill Counting — All-in-One", page_icon="💊", layout="wide")

# ---------- Helpers ----------
def bytes_to_npimg(bytes_data):
    arr = np.frombuffer(bytes_data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def resize_for_display(img, max_w=1400, max_h=1000):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(w*s), int(h*s))) if s < 1.0 else img

def preprocess(img_bgr, blur=3, clahe=True):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        L = c.apply(L)
    lab = cv2.merge([L, A, B])
    work = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    if blur > 0:
        gray = cv2.medianBlur(gray, blur)
    return work, gray

def annotate(base_img, contours=None, ellipses=None, masks=None, heatmap=None, keypoints=None):
    vis = base_img.copy()
    if contours:
        cv2.drawContours(vis, contours, -1, (80,220,80), 2)
    if ellipses:
        for e in ellipses:
            (x,y),(MA,ma),ang = e
            cv2.ellipse(vis, (int(x),int(y)), (int(MA/2),int(ma/2)), ang, 0, 360, (220,80,220), 2)
    if masks:
        overlay = np.zeros_like(vis)
        for m, conf in masks:
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, (0,180,255), 2)
        vis = cv2.addWeighted(vis, 1.0, overlay, 0.6, 0)
    if heatmap is not None:
        hm_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        vis = cv2.addWeighted(vis, 0.6, hm_color, 0.4, 0)
    if keypoints:
        for (x,y) in keypoints:
            cv2.circle(vis, (int(x),int(y)), 6, (50,180,255), 2)
    return vis

def save_img(img):
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"annotated_{ts}.png"
    cv2.imwrite(str(path), img)
    return str(path)

def export_session(rows):
    if not rows: return None, None
    df = pd.DataFrame(rows)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"counts_{ts}.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path), df

def build_conf_heatmap(shape_img, masks):
    if not masks: return None
    acc = np.zeros(shape_img[:2], dtype=np.float32)
    for m, conf in masks:
        acc += (m.astype(np.float32)/255.0) * float(conf)
    acc = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return acc

def barcode_from_image(img_bgr):
    if not BARCODE_AVAILABLE: return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    try:
        codes = zbar_decode(gray)
        if not codes: return None
        return codes[0].data.decode("utf-8", errors="ignore")
    except Exception:
        return None

# ---------- Classical detectors ----------
def detect_contours(gray, min_area=100, max_area=100000, min_circ=0.2):
    thr = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV,35,7)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), 1)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep, ellipses = [], []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area: continue
        peri = cv2.arcLength(c, True) + 1e-6
        circ = 4*math.pi*area/(peri*peri)
        if circ < min_circ: continue
        keep.append(c)
        if len(c) >= 5:
            ellipses.append(cv2.fitEllipse(c))
    return keep, ellipses, thr

def detect_watershed(gray, min_area=100, max_area=100000):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = morphology.opening(bw > 0, morphology.disk(2))
    bw = (bw*255).astype(np.uint8)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    local_max = peak_local_max(dist, labels=bw, min_distance=15, exclude_border=False)
    markers = np.zeros_like(bw, dtype=np.int32)
    for i,(y,x) in enumerate(local_max, start=1):
        markers[y,x] = i
    labels = watershed(-dist, markers, mask=bw)
    contours, ellipses = [], []
    for label in np.unique(labels):
        if label == 0: continue
        mask = (labels==label).astype(np.uint8)*255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < min_area or area > max_area: continue
        contours.append(c)
        if len(c) >= 5:
            ellipses.append(cv2.fitEllipse(c))
    return contours, ellipses, bw

# ---------- YOLO segmentation ----------
def yolo_detect(img_bgr, model, conf_thr=0.25):
    """
    Returns: list of (mask_uint8, confidence), count
    If model has masks -> use them; otherwise box -> filled rectangle mask.
    """
    res = model.predict(img_bgr, conf=conf_thr, verbose=False)
    masks_out, count = [], 0
    for r in res:
        if hasattr(r, "masks") and r.masks is not None and r.masks.data is not None:
            for m, c in zip(r.masks.data.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                m = (m*255).astype(np.uint8)
                # ensure mask matches image size
                m = cv2.resize(m, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                masks_out.append((m, float(c)))
                count += 1
        else:
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                x1,y1,x2,y2 = map(int, b)
                m = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
                cv2.rectangle(m, (x1,y1), (x2,y2), 255, -1)
                masks_out.append((m, float(c)))
                count += 1
    return masks_out, count

# ---------- Google Sheets (via Secrets) ----------
def materialize_creds_from_secrets():
    js = st.secrets.get("GOOGLE_APPLICATION_CREDENTIALS_TEXT", None)
    if not js:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.write(js.encode("utf-8")); tmp.flush()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
    return tmp.name

def get_sheet_client():
    if not SHEETS_AVAILABLE:
        return None
    creds_path = materialize_creds_from_secrets()
    if not creds_path or not Path(creds_path).exists():
        return None
    scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    return gspread.authorize(creds)

def append_to_sheet(sheet_id, sheet_tab, row_values):
    try:
        client = get_sheet_client()
        if not client: return False, "No service account / secrets not set"
        sh = client.open_by_key(sheet_id)
        ws = sh.worksheet(sheet_tab)
        ws.append_row(row_values, value_input_option="USER_ENTERED")
        return True, "OK"
    except Exception as e:
        return False, str(e)

# ---------- UI: Mobile Mode ----------
def mobile_mode_default():
    # Default ON for phone; you can toggle in sidebar.
    return st.session_state.get("mobile_mode", True)

with st.sidebar:
    st.subheader("📱 Mobile Mode")
    st.session_state["mobile_mode"] = st.toggle("Optimized for phone", value=True,
        help="Bigger targets, AI + Double-check ON by default.")

# ---------- Sidebar controls ----------
use_ai_default       = True if mobile_mode_default() else (YOLO_AVAILABLE and DEFAULT_MODEL_PATH.exists())
double_check_default = True
show_heatmap_default = True

st.title(APP_NAME)
st.caption("Camera/Photo → AI YOLO-seg + Classical, QC double-check, scatter risk, heatmap, barcode, Sheets sync")

with st.sidebar:
    st.header("Detection")
    use_ai = st.checkbox("Enable AI model (YOLOv8-seg)", value=use_ai_default,
                         help="If weights missing, app falls back to classical.")
    model_path = st.text_input("Model path (.pt)", value=str(DEFAULT_MODEL_PATH))
    uploaded_model = st.file_uploader("Upload model (.pt)", type=["pt"], help="Optional, load weights from your phone")
    if uploaded_model is not None:
        # save uploaded weights to a temp file
        tmp_pt = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        tmp_pt.write(uploaded_model.read()); tmp_pt.flush()
        model_path = tmp_pt.name
        st.success("Model loaded from upload.")

    conf_thr = st.slider("AI confidence threshold", 0.05, 0.95, 0.25, 0.05)

    st.subheader("Classical (fallback)")
    detector = st.radio("Detector", ["Watershed (touching pills)", "Contour (general)", "Hough (round only)"])
    min_area = st.slider("Min area (px)", 20, 5000, 150)
    max_area = st.slider("Max area (px)", 1000, 300000, 80000)
    min_circ = st.slider("Min circularity (Contour)", 0.00, 1.00, 0.20, 0.01)
    blur_amt = st.slider("Preprocess blur", 0, 9, 3, 1)
    do_clahe = st.checkbox("CLAHE lighting normalize", value=True)

    st.header("Quality & Safety")
    double_check = st.checkbox("Double-check (AI vs classical)", value=double_check_default)
    dc_tolerance = st.slider("Tolerance (± count)", 0, 10, 2)
    scatter_warn = st.checkbox("Scatter-risk detector", value=True,
                               help="Warn if tiny fragments percentage is high.")
    scatter_area = st.slider("Tiny fragment size ≤ (px)", 5, 500, 40)
    scatter_ratio = st.slider("Warn if tiny % ≥", 1, 50, 12)

    st.header("Heatmap & Barcode")
    show_heatmap = st.checkbox("Show confidence heatmap (AI)", value=show_heatmap_default)
    barcode_enable = st.checkbox("Barcode tie-in (UPC/EAN/DIN)", value=True)

    st.header("Logging")
    batch_id = st.text_input("Batch / Rx / Note", value="")
    staff = st.text_input("Staff", value="")
    location = st.text_input("Location", value="")
    manual_adjust = st.number_input("Manual adjust (±)", value=0, step=1)

    st.header("Google Sheets (optional)")
    sheets_on = st.checkbox("Sync to Sheets", value=False)
    sheet_id = st.text_input("Sheet ID", value=st.secrets.get("SHEET_ID",""))
    sheet_tab = st.text_input("Sheet tab", value=st.secrets.get("SHEET_TAB","Counts"))

# lazy-load YOLO if enabled
yolo_model = None
if use_ai and YOLO_AVAILABLE and Path(model_path).exists():
    try:
        yolo_model = YOLO(model_path)
    except Exception as e:
        st.sidebar.warning(f"Could not load model: {e}")
        yolo_model = None
elif use_ai and not YOLO_AVAILABLE:
    st.sidebar.info("Ultralytics not installed; AI disabled.")
elif use_ai and not Path(model_path).exists():
    st.sidebar.info("Model file not found; AI disabled.")

session_rows = st.session_state.setdefault("rows", [])

# ---------- Core processing ----------
def classical_count(img_bgr):
    work, gray = preprocess(img_bgr, blur=blur_amt, clahe=do_clahe)
    if detector.startswith("Watershed"):
        contours, ellipses, dbg = detect_watershed(gray, min_area=min_area, max_area=max_area)
        vis = annotate(img_bgr, contours=contours, ellipses=ellipses)
        return len(contours), vis, {"contours": contours, "ellipses": ellipses, "debug": dbg}
    elif detector.startswith("Contour"):
        contours, ellipses, dbg = detect_contours(gray, min_area=min_area, max_area=max_area, min_circ=min_circ)
        vis = annotate(img_bgr, contours=contours, ellipses=ellipses)
        return len(contours), vis, {"contours": contours, "ellipses": ellipses, "debug": dbg}
    else:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                   param1=100, param2=20, minRadius=8, maxRadius=120)
        cnts, ell = [], []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x,y,r) in circles[0, :]:
                cnt = cv2.ellipse2Poly((int(x),int(y)), (int(r),int(r)), 0, 0, 360, 10)
                cnts.append(cnt)
                ell.append(((x,y), (2*r,2*r), 0))
        vis = annotate(img_bgr, contours=cnts, ellipses=ell)
        return len(cnts), vis, {"contours": cnts, "ellipses": ell, "debug": None}

def scatter_risk_metric(binary_like, area_px=40):
    if binary_like is None: return 0, 0
    cnts, _ = cv2.findContours((binary_like > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tiny = [c for c in cnts if cv2.contourArea(c) <= area_px]
    return len(tiny), len(cnts)

def process_image(img_bgr, title=""):
    # AI pass
    ai_count = None; masks = None; heatmap = None
    if yolo_model is not None:
        try:
            masks, ai_raw = yolo_detect(img_bgr, yolo_model, conf_thr=conf_thr)
            ai_count = ai_raw
            if show_heatmap:
                heatmap = build_conf_heatmap(img_bgr, masks)
        except Exception as e:
            st.warning(f"AI inference failed: {e}")

    # Classical pass (always available)
    cls_count, cls_vis, cls_dbg = classical_count(img_bgr)

    # Double-check QC
    qc_flag = False; qc_msg = "OK"
    if double_check and ai_count is not None:
        if abs(ai_count - cls_count) > dc_tolerance:
            qc_flag = True
            qc_msg = f"DISAGREE: AI={ai_count} vs Classical={cls_count} (±{dc_tolerance})"

    # Choose primary count
    primary = ai_count if (ai_count is not None) else cls_count
    count_final = int((primary or 0) + int(manual_adjust))

    # Visualization
    base_small = resize_for_display(img_bgr)
    if ai_count is not None and masks is not None:
        disp_masks = [(cv2.resize(m, (base_small.shape[1], base_small.shape[0]), interpolation=cv2.INTER_NEAREST), conf)
                      for (m,conf) in masks]
        hm_small = cv2.resize(heatmap, (base_small.shape[1], base_small.shape[0]), interpolation=cv2.INTER_NEAREST) if heatmap is not None else None
        vis = annotate(base_small, masks=disp_masks, heatmap=hm_small)
    else:
        # Draw classical contours on resized image
        scale_x = base_small.shape[1] / img_bgr.shape[1]
        scale_y = base_small.shape[0] / img_bgr.shape[0]
        scaled_cnts = []
        for c in (cls_dbg["contours"] or []):
            c_scaled = (c * [scale_x, scale_y]).astype(np.int32)
            scaled_cnts.append(c_scaled)
        vis = annotate(base_small, contours=scaled_cnts)

    # Scatter risk from classical binary
    warn_txt = None
    if scatter_warn:
        tiny, total = scatter_risk_metric(cls_dbg["debug"], area_px=scatter_area)
        ratio = (100.0 * tiny / max(total, 1))
        if ratio >= scatter_ratio:
            warn_txt = f"Scatter risk: {ratio:.1f}% tiny fragments (≥{scatter_ratio}%)."
            st.warning(warn_txt)

    # Barcode
    code = barcode_from_image(img_bgr) if barcode_enable else None

    # Save annotated image
    out_path = save_img(vis)

    # Log row
    row = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "mode_primary": "AI" if (ai_count is not None) else "Classical",
        "ai_count": ai_count if ai_count is not None else "",
        "classical_count": cls_count,
        "manual_adjust": int(manual_adjust),
        "count_final": count_final,
        "qc": "FLAG" if qc_flag else "OK",
        "qc_msg": qc_msg,
        "scatter_warn": warn_txt or "",
        "barcode": code or "",
        "batch": batch_id,
        "staff": staff,
        "location": location,
        "image_path": out_path
    }
    st.session_state["rows"].append(row)

    # Show
    header = f"{title} • Final: {count_final}  |  "
    if ai_count is not None: header += f"AI {ai_count} • "
    header += f"Classical {cls_count}  |  Adj {int(manual_adjust)}"
    st.image(vis, caption=header, use_column_width=True)

    c1, c2, c3 = st.columns(3)
    with c1: st.json({"QC": row["qc"], "msg": row["qc_msg"]})
    with c2: st.json({"barcode": code or "(none)"})
    with c3: st.write(f"Saved → `{out_path}`")

    # Optional Google Sheets sync
    if sheets_on and sheet_id:
        values = [
            row["timestamp"], row["mode_primary"], row["ai_count"], row["classical_count"],
            row["manual_adjust"], row["count_final"], row["qc"], row["qc_msg"],
            row["scatter_warn"], row["barcode"], row["batch"], row["staff"], row["location"], row["image_path"]
        ]
        ok, msg = append_to_sheet(sheet_id, sheet_tab or "Counts", values)
        if ok: st.success("Synced to Google Sheets.")
        else:  st.warning(f"Sheets sync failed: {msg}")

    return row

# ---------- Tabs ----------
tab_cam, tab_photo, tab_hist = st.tabs(["📷 Camera", "🖼️ Photo Upload", "📑 History / Exports"])

with tab_cam:
    st.markdown("**Tip:** Use a high-contrast background or our printed reference mat. Turn on flash if needed.")
    cam = st.camera_input("Camera (capture a frame)")
    if cam is not None:
        img = bytes_to_npimg(cam.getvalue())
        process_image(img, "Camera")

with tab_photo:
    up = st.file_uploader("Upload a photo (JPG/PNG)", type=["jpg","jpeg","png"])
    if up:
        img = bytes_to_npimg(up.read())
        process_image(img, "Photo")

with tab_hist:
    rows = st.session_state.get("rows", [])
    if rows:
        csv_path, df = export_session(rows)
        st.dataframe(df, use_container_width=True)
        if df is not None:
            st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                               file_name="pill_counts_session.csv", mime="text/csv")
            st.caption(f"Session CSV also saved to: {csv_path}")
    else:
        st.info("No history yet. Capture from Camera or Upload a Photo to start.")
