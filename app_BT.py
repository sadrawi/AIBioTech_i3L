import math
import statistics
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent
LOGO_PATH = BASE_DIR / "i3LUniversity.png"
GRAM_MODEL_PATH = BASE_DIR / "best_gramstain.pt"
COLONY_MODEL_PATH = BASE_DIR / "best_colony.pt"
AMR_MODEL_PATH = BASE_DIR / "best_amr.pt"
NONE_OPTION = "(none)"

st.set_page_config(
    page_title="i3L AI System",
    layout="wide",
    initial_sidebar_state="auto",
)


@st.cache_resource
def load_model(path):
    return YOLO(str(path))


def get_model(path):
    if not path.exists():
        st.error(f"Model file not found: {path.name}. Upload it to the repo root.")
        return None
    return load_model(path)


def read_image(uploaded_file):
    return Image.open(uploaded_file).convert("RGB")


def plot_result(result, show_labels=True, line_width=None):
    plotted_bgr = result.plot(labels=show_labels, conf=show_labels, line_width=line_width)
    return plotted_bgr[:, :, ::-1].copy()


def get_boxes(result):
    if result.boxes is not None:
        return result.boxes
    return getattr(result, "obb", None)


def class_probabilities(result, top_k=5):
    probs = result.probs.data.tolist()
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    rows = []
    for i in order[:top_k]:
        rows.append({"Class": result.names[i], "Probability": f"{probs[i]:.1%}"})
    return pd.DataFrame(rows)


def count_per_class(result):
    boxes = get_boxes(result)
    if boxes is None:
        return pd.DataFrame()
    names = result.names
    counts = {}
    for cls_id in boxes.cls.tolist():
        name = names[int(cls_id)]
        if name in counts:
            counts[name] += 1
        else:
            counts[name] = 1
    rows = []
    for name in sorted(counts):
        rows.append({"Class": name, "Count": counts[name]})
    return pd.DataFrame(rows)


def sci_latex(value):
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10 ** exponent)
    return f"{mantissa:.2f} \\times 10^{{{exponent}}}"


def show_side_by_side(original, detected):
    col1, col2 = st.columns(2)
    col1.image(original, caption="Original", width="stretch")
    col2.image(detected, caption="Detected", width="stretch")


# ---------- Gram staining ----------

def gram_stain_tab():
    st.subheader("Gram Staining Detection")
    model = get_model(GRAM_MODEL_PATH)
    if model is None:
        return

    is_classifier = model.task == "classify"
    conf = 0.25
    if not is_classifier:
        conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05, key="gram_conf")
    source_img = st.file_uploader(
        "Choose a Gram-stained image...",
        type=("jpg", "jpeg", "png"),
        key="gram_upload",
    )

    if st.button("Detect", key="gram_detect"):
        if source_img is None:
            st.warning("Please upload an image first.")
            return

        img = read_image(source_img)
        result = model.predict(img, conf=conf)[0]
        show_side_by_side(img, plot_result(result))

        if result.probs is not None:
            top_class = result.names[result.probs.top1]
            top_conf = result.probs.top1conf.item()
            st.success(f"Predicted class: **{top_class}** ({top_conf:.1%})")
            st.dataframe(class_probabilities(result), hide_index=True)
            return

        counts = count_per_class(result)
        if counts.empty:
            st.info("No objects detected. Try lowering the confidence threshold.")
        else:
            st.dataframe(counts, hide_index=True)


# ---------- Colony counting ----------

def colony_tab():
    st.subheader("Colony Counting")
    model = get_model(COLONY_MODEL_PATH)
    if model is None:
        return
    if model.task == "classify":
        st.error(f"{COLONY_MODEL_PATH.name} is a classification model. This tab needs an object detection model.")
        return

    col_a, col_b, col_c = st.columns(3)
    conf = col_a.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05, key="colony_conf")
    imgsz = col_b.selectbox("Inference size (px)", (640, 1024, 1280), key="colony_imgsz")
    show_labels = col_c.checkbox("Show labels on boxes", value=False, key="colony_labels")

    with st.expander("CFU/mL calculation (optional)"):
        dilution_exp = st.number_input(
            "Dilution exponent $n$ (plated from the $10^{-n}$ dilution)",
            min_value=0,
            max_value=12,
            value=0,
            step=1,
            key="colony_dilution",
        )
        volume_ml = st.number_input(
            "Volume plated $V$ (mL)",
            min_value=0.001,
            value=0.1,
            step=0.05,
            format="%.3f",
            key="colony_volume",
        )

    source_img = st.file_uploader(
        "Choose a plate image...",
        type=("jpg", "jpeg", "png"),
        key="colony_upload",
    )

    if st.button("Count colonies", key="colony_detect"):
        if source_img is None:
            st.warning("Please upload an image first.")
            return

        img = read_image(source_img)
        result = model.predict(img, conf=conf, imgsz=imgsz, max_det=1000)[0]
        show_side_by_side(img, plot_result(result, show_labels=show_labels, line_width=1))

        n_colonies = len(get_boxes(result))
        st.metric("Total colonies", n_colonies)

        counts = count_per_class(result)
        if counts.empty:
            st.info("No colonies detected. Try lowering the confidence threshold.")
            return
        st.dataframe(counts, hide_index=True)

        cfu = n_colonies * (10 ** dilution_exp) / volume_ml
        st.latex(r"\text{CFU/mL} = \frac{N \times 10^{n}}{V}")
        st.latex(
            rf"\text{{CFU/mL}} = \frac{{{n_colonies} \times 10^{{{dilution_exp}}}}}{{{volume_ml:g}}} = {sci_latex(cfu)}"
        )

        if n_colonies < 30 or n_colonies > 300:
            st.warning(
                "Count is outside the 30–300 countable range, so the CFU/mL estimate is less reliable. "
                "Use a plate from a different dilution if available."
            )


# ---------- Antimicrobial resistance (disk diffusion) ----------

def class_options(model):
    options = [NONE_OPTION]
    for cls_id in sorted(model.names):
        options.append(model.names[cls_id])
    return options


def guess_index(options, keywords):
    for i, name in enumerate(options):
        lower = name.lower()
        for kw in keywords:
            if kw in lower:
                return i
    return 0


def select_boxes(result, class_name):
    selected = []
    if class_name == NONE_OPTION:
        return selected
    boxes = get_boxes(result)
    if boxes is None:
        return selected
    names = result.names
    xyxy = boxes.xyxy.tolist()
    cls_ids = boxes.cls.tolist()
    confs = boxes.conf.tolist()
    for i in range(len(cls_ids)):
        if names[int(cls_ids[i])] != class_name:
            continue
        x1, y1, x2, y2 = xyxy[i]
        selected.append({
            "xyxy": (x1, y1, x2, y2),
            "d_px": ((x2 - x1) + (y2 - y1)) / 2,
            "conf": confs[i],
        })
    return selected


def compute_scale(disks, disk_mm, manual_scale):
    if manual_scale > 0:
        latex = rf"s = {manual_scale:.4f}\,\text{{mm/px}} \quad \text{{(manual)}}"
        return manual_scale, latex
    if not disks:
        return None, None
    diameters = []
    for disk in disks:
        diameters.append(disk["d_px"])
    median_px = statistics.median(diameters)
    scale = disk_mm / median_px
    latex = (
        rf"s = \frac{{d_{{\text{{disk}}}}}}{{\operatorname{{median}}\left(d_{{\text{{disk,px}}}}\right)}}"
        rf" = \frac{{{disk_mm:g}\,\text{{mm}}}}{{{median_px:.1f}\,\text{{px}}}}"
        rf" = {scale:.4f}\,\text{{mm/px}}"
    )
    return scale, latex


def draw_zone_ids(image, zones):
    out = image.copy()
    font_scale = max(0.6, out.shape[1] / 1200)
    thickness = max(1, round(font_scale * 2))
    for i, zone in enumerate(zones):
        x1, y1, x2, y2 = zone["xyxy"]
        org = (int(x1) + 4, int(y2) - 8)
        text = f"Z{i + 1}"
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness, cv2.LINE_AA)
    return out


def interpret_zone(d_mm, s_bp, r_bp):
    if pd.isna(s_bp) or pd.isna(r_bp):
        return "—"
    if d_mm >= s_bp:
        return "S"
    if d_mm <= r_bp:
        return "R"
    return "I"


def show_amr_output(output):
    show_side_by_side(output["image"], output["plotted"])

    if output["counts"].empty:
        st.info("No objects detected. Try lowering the confidence threshold.")
        return
    st.dataframe(output["counts"], hide_index=True)

    zones = output["zones"]
    if not zones:
        if output["zone_class"] != NONE_OPTION:
            st.info("No inhibition zones detected.")
        return

    if output["scale"] is None:
        st.info(
            "Scale unknown: select the antibiotic disk class or enter a manual scale "
            "to convert zone diameters to mm."
        )
        rows = []
        for i, zone in enumerate(zones):
            rows.append({
                "Zone": f"Z{i + 1}",
                "Diameter (px)": round(zone["d_px"], 1),
                "Confidence": round(zone["conf"], 2),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        return

    st.latex(output["scale_latex"])
    st.latex(r"d_{\text{zone}} = s \times d_{\text{zone,px}}")

    rows = []
    for i, zone in enumerate(zones):
        rows.append({
            "Zone": f"Z{i + 1}",
            "Diameter (mm)": round(zone["d_px"] * output["scale"], 1),
            "Antibiotic": "",
            "S ≥ (mm)": math.nan,
            "R ≤ (mm)": math.nan,
        })

    st.markdown("**Enter the antibiotic and its breakpoints for each zone:**")
    edited = st.data_editor(
        pd.DataFrame(rows),
        hide_index=True,
        disabled=["Zone", "Diameter (mm)"],
        column_config={
            "Antibiotic": st.column_config.TextColumn("Antibiotic"),
            "S ≥ (mm)": st.column_config.NumberColumn("S ≥ (mm)", min_value=0.0, step=1.0),
            "R ≤ (mm)": st.column_config.NumberColumn("R ≤ (mm)", min_value=0.0, step=1.0),
        },
        key=f"amr_editor_{output['run_id']}",
    )

    result_rows = []
    for i in range(len(edited)):
        row = edited.iloc[i]
        result_rows.append({
            "Zone": row["Zone"],
            "Antibiotic": row["Antibiotic"],
            "Diameter (mm)": row["Diameter (mm)"],
            "Interpretation": interpret_zone(row["Diameter (mm)"], row["S ≥ (mm)"], row["R ≤ (mm)"]),
        })
    st.dataframe(pd.DataFrame(result_rows), hide_index=True)
    st.caption(
        "S = susceptible, I = intermediate, R = resistant. Use breakpoints from the current CLSI M100 "
        "or EUCAST table for your organism–antibiotic pair. For EUCAST \"R < y\", enter y − 1 in the R column. "
        "Zones cut off by the plate edge or merged with a neighbour will be under- or over-estimated."
    )


def amr_tab():
    st.subheader("Antimicrobial Resistance Detection (Disk Diffusion)")
    model = get_model(AMR_MODEL_PATH)
    if model is None:
        return
    if model.task == "classify":
        st.error(f"{AMR_MODEL_PATH.name} is a classification model. This tab needs an object detection model.")
        return

    options = class_options(model)

    col_a, col_b = st.columns(2)
    conf = col_a.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05, key="amr_conf")
    show_labels = col_b.checkbox("Show labels on boxes", value=True, key="amr_labels")

    with st.expander("Zone measurement settings"):
        col_c, col_d = st.columns(2)
        disk_class = col_c.selectbox(
            "Antibiotic disk class",
            options,
            index=guess_index(options, ("disk", "disc")),
            key="amr_disk_class",
        )
        zone_class = col_d.selectbox(
            "Inhibition zone class",
            options,
            index=guess_index(options, ("zone", "inhib", "halo")),
            key="amr_zone_class",
        )
        col_e, col_f = st.columns(2)
        disk_mm = col_e.number_input(
            "Disk diameter (mm)",
            min_value=1.0,
            value=6.0,
            step=0.5,
            key="amr_disk_mm",
        )
        manual_scale = col_f.number_input(
            "Manual scale (mm/pixel), 0 = calibrate from disks",
            min_value=0.0,
            value=0.0,
            step=0.001,
            format="%.4f",
            key="amr_scale",
        )

    source_img = st.file_uploader(
        "Choose a disk diffusion plate image...",
        type=("jpg", "jpeg", "png"),
        key="amr_upload",
    )

    if st.button("Detect", key="amr_detect"):
        if source_img is None:
            st.warning("Please upload an image first.")
        else:
            img = read_image(source_img)
            result = model.predict(img, conf=conf)[0]
            disks = select_boxes(result, disk_class)
            zones = select_boxes(result, zone_class)
            scale, scale_latex = compute_scale(disks, disk_mm, manual_scale)
            plotted = plot_result(result, show_labels=show_labels)
            if zones:
                plotted = draw_zone_ids(plotted, zones)
            run_id = st.session_state.get("amr_run_id", 0) + 1
            st.session_state["amr_run_id"] = run_id
            st.session_state["amr_output"] = {
                "run_id": run_id,
                "image": img,
                "plotted": plotted,
                "counts": count_per_class(result),
                "zones": zones,
                "zone_class": zone_class,
                "scale": scale,
                "scale_latex": scale_latex,
            }

    output = st.session_state.get("amr_output")
    if output is not None:
        show_amr_output(output)


# ---------- Layout ----------

if LOGO_PATH.exists():
    st.image(Image.open(LOGO_PATH))
st.markdown(
    "<h1 style='text-align: center;'>i3L AI-based Microbiology Detection</h1>",
    unsafe_allow_html=True,
)

tab_gram, tab_colony, tab_amr = st.tabs(
    ["Gram Staining Detection", "Colony Counting", "Antimicrobial Resistance"]
)

with tab_gram:
    gram_stain_tab()

with tab_colony:
    colony_tab()

with tab_amr:
    amr_tab()
