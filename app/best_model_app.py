import tempfile
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_utils import load_checkpoint_model, predict_audio_path


st.set_page_config(page_title="AI Music Detection Demo", layout="wide")
st.title("AI Music Detection Demo")
st.write(
    "Bu uygulama mevcut en iyi checkpoint'i kullanarak yuklenen ses dosyasi icin "
    "track-level fake olasiligi uretir."
)


@st.cache_resource
def get_model_bundle(model_name: str, checkpoint_path: str):
    return load_checkpoint_model(model_name, Path(checkpoint_path))


def render_spectrogram(spec: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
    ax.set_title("Log Spectrogram")
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Frequency Bins")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    st.pyplot(fig, clear_figure=True)


default_checkpoint = "models/best_resnet_cropped.pth"
model_name = st.sidebar.selectbox(
    "Model",
    options=["resnet", "artifactnet", "transformer"],
    index=0,
)
checkpoint_path = st.sidebar.text_input("Checkpoint", value=default_checkpoint)

uploaded_file = st.file_uploader("Bir audio dosyasi yukleyin", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.read())
        temp_path = Path(handle.name)

    try:
        model, device = get_model_bundle(model_name, checkpoint_path)
        result = predict_audio_path(model, device, temp_path)

        prob_fake = float(result["track_prob_fake"])
        label_text = "Fake / transformed" if result["track_pred"] == 1 else "Real"

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Tahmin", label_text)
        metric_col2.metric("Fake Olasiligi", f"{prob_fake:.4f}")
        metric_col3.metric("Crop Sayisi", str(result["n_crops"]))

        st.progress(min(max(prob_fake, 0.0), 1.0), text="Fake olasiligi")

        left_col, right_col = st.columns([2, 1])
        with left_col:
            render_spectrogram(result["spectrogram"])
        with right_col:
            crop_probs = np.array(result["crop_probs"], dtype=float)
            st.subheader("Crop-level Ozet")
            st.write(f"Ortalama: `{crop_probs.mean():.4f}`")
            st.write(f"Maksimum: `{crop_probs.max():.4f}`")
            st.write(f"Minimum: `{crop_probs.min():.4f}`")
            st.line_chart(crop_probs)

    except Exception as exc:
        st.error(f"Tahmin sirasinda hata olustu: {exc}")
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
else:
    st.info("Demo icin bir ses dosyasi yukleyin.")
