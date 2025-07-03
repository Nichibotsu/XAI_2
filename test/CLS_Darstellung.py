import streamlit as st
import torch
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import os
from pathlib import Path

# ----------- Modell laden -----------

@st.cache_resource
def load_model():
    model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model.eval()
    return model, processor

model, processor = load_model()

# ----------- Helper zum Laden der Bilder -----------

def load_images_from_folder(base_path, max_per_class=30):
    image_paths = []
    labels = []

    base = Path(base_path)
    for class_dir in base.iterdir():
        if class_dir.is_dir():
            files = list(class_dir.glob("*.jpg"))[:max_per_class]
            image_paths.extend(files)
            labels.extend([class_dir.name] * len(files))
    return image_paths, labels

# ----------- UI -----------

st.title("🐶🐱 CLS-Repräsentationen – Oxford Pets oder eigener Datensatz")

base_dir = st.text_input("📁 Gib den Pfad zu deinem Bilderordner ein", value="data/pets_subset_mini")
max_imgs = st.slider("🔢 Max. Bilder pro Klasse", 5, 50, 20)

# Zeige Inhalt optional
if st.checkbox("📂 Zeige Unterordner"):
    try:
        st.write([f.name for f in Path(base_dir).iterdir() if f.is_dir()])
    except:
        st.warning("Pfad ungültig oder nicht gefunden.")

image_paths, labels = load_images_from_folder(base_dir, max_imgs)


if len(image_paths) == 0:
    st.warning("Keine Bilder gefunden. Stelle sicher, dass dein Pfad korrekt ist.")
else:
    st.write(f"📸 {len(image_paths)} Bilder aus {len(set(labels))} Klassen geladen.")

    cls_vectors = []

    progress = st.progress(0)
    for i, img_path in enumerate(image_paths):
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

        cls_vectors.append(cls)
        progress.progress((i + 1) / len(image_paths))

    cls_array = np.stack(cls_vectors)

    # ----------- Reduktion -----------

    method = st.selectbox("Reduktionsmethode", ["PCA", "t-SNE"])

    if method == "PCA":
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, perplexity=30, init="random", random_state=42)

    reduced = reducer.fit_transform(cls_array)

    # ----------- Plotly Scatter -----------

    fig = px.scatter(
        x=reduced[:, 0],
        y=reduced[:, 1],
        color=labels,
        hover_name=[p.name for p in image_paths],
        title=f"{method} Projektion der CLS-Repräsentationen",
        labels={"x": "Dim 1", "y": "Dim 2"},
        height=700
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color='DarkSlateGrey')))
    st.plotly_chart(fig)

    # ----------- Optional: Vektoren zeigen -----------

    with st.expander("📊 CLS-Vektoren"):
        for name, vec in zip(image_paths, cls_vectors):
            st.write(f"🔹 **{name.name}**")
            st.write(vec)
