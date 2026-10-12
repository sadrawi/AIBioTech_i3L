import math
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent
LOGO_PATH = BASE_DIR / "i3LUniversity.png"
GRAM_MODEL_PATH = BASE_DIR / "best_gramstain.pt"
COLONY_MODEL_PATH = BASE_DIR / "best_colony.pt"

col00, col01 = st.columns([0.85,0.55])
with col01:
    st.image(BASE_DIR / "BT v1.png")
with col00:
    st.image(BASE_DIR / "SHL.png")

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


def count_per_class(result):
    names = result.names
    counts = {}
    for cls_id in result.boxes.cls.tolist():
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


def gram_stain_tab():
    st.subheader("Gram Staining Detection")
    model = get_model(GRAM_MODEL_PATH)
    if model is None:
        return

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

        counts = count_per_class(result)
        if counts.empty:
            st.info("No objects detected. Try lowering the confidence threshold.")
        else:
            st.dataframe(counts, hide_index=True)


def colony_tab():
    st.subheader("Colony Counting")
    model = get_model(COLONY_MODEL_PATH)
    if model is None:
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

        n_colonies = len(result.boxes)
        st.metric("Total colonies", n_colonies)

        counts = count_per_class(result)
        if counts.empty:
            st.info("No colonies detected. Try lowering the confidence threshold.")
            return
        st.dataframe(counts, hide_index=True)

        # cfu = n_colonies * (10 ** dilution_exp) / volume_ml
        # st.latex(r"\text{CFU/mL} = \frac{N \times 10^{n}}{V}")
        # st.latex(
        #     rf"\text{{CFU/mL}} = \frac{{{n_colonies} \times 10^{{{dilution_exp}}}}}{{{volume_ml:g}}} = {sci_latex(cfu)}"
        # )

        # if n_colonies < 30 or n_colonies > 300:
        #     st.warning(
        #         "Count is outside the 30–300 countable range, so the CFU/mL estimate is less reliable. "
        #         "Use a plate from a different dilution if available."
        #     )


if LOGO_PATH.exists():
    st.image(Image.open(LOGO_PATH))
st.title("i3L AI-based Microbiology Detection")

tab_gram, tab_colony = st.tabs(["Gram Staining Detection", "Colony Counting"])

with tab_gram:
    gram_stain_tab()

with tab_colony:
    colony_tab()
