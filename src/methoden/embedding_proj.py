from typing import List, Sequence
import streamlit as st
import torch
import numpy as np
from PIL import Image
import plotly.express as px
import torchvision.transforms as T
from sklearn.manifold import TSNE, MDS
from cuml.manifold import TSNE as cuTSNE
from cuml.manifold import UMAP as cuUMAP
from umap import UMAP
from sklearn.decomposition import PCA
import uuid
import pandas as pd


def _get_layer(model: torch.nn.Module, layer_path: str):
    layer = model
    for attr in layer_path.split('.'):
        layer = getattr(layer, attr)
    return layer


def _collect_activations(model: torch.nn.Module, images: List[Image.Image], layer_name: str, device: str = "cuda", batch_size: int = 32) -> np.ndarray:
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    acts: List[torch.Tensor] = []

    def hook(_, __, output):
        pooled = (output.mean((2, 3)) if output.ndim == 4 else output)
        acts.append(pooled.detach().cpu())

    handle = _get_layer(model, layer_name).register_forward_hook(hook)

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    progress = st.progress(0.0, text="Bilder durch Netzwerk …")
    log_box  = st.empty()  # Text‑Placeholder für Debug‑Info in UI

    total = len(images)
    processed = 0
    for i in range(0, total, batch_size):
        batch_imgs = torch.stack([transform(img) for img in images[i:i + batch_size]]).to(device)
        with torch.no_grad():
            model(batch_imgs)
        processed += batch_imgs.size(0)

        frac = processed / total
        progress.progress(frac)
        msg = f"Aktivierungen gesammelt: {processed}/{total}"
        log_box.text(msg)
        print(msg, flush=True)  # Debug in Terminal/Logs

    handle.remove()
    progress.empty()
    log_box.empty()

    return torch.cat(acts).numpy()


def _reduce(vecs: np.ndarray, method: str = "umap", dim: int = 2, params: dict = None):
    params = params or {}
    vecs32 = vecs.astype(np.float32)

    if method.lower() == "umap":
        try:
            reducer = cuUMAP(n_components=dim, **params)
            emb = reducer.fit_transform(vecs32)
            return emb if isinstance(emb, np.ndarray) else emb.get()
        except Exception:
            reducer = UMAP(n_components=dim, **params)
            return reducer.fit_transform(vecs32)

    elif method.lower() == "tsne":
        try:
            reducer = cuTSNE(n_components=dim, **params)
            emb = reducer.fit_transform(vecs32)
            return emb.get()
        except Exception:
            reducer = TSNE(n_components=dim, **params)
            return reducer.fit_transform(vecs32)

    elif method.lower() == "mds":
        reducer = MDS(n_components=dim, **params)
        return reducer.fit_transform(vecs32)

    else:  # PCA
        reducer = PCA(n_components=dim)
        return reducer.fit_transform(vecs32)

# Visualisierung 

def _plot(emb: np.ndarray, labels: Sequence[str], dim: int):
    df = pd.DataFrame(emb, columns=["x", "y"] + (["z"] if dim == 3 else []))
    df["label"] = labels

    if dim == 3:
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="label", width=700, height=1000)
        fig.update_traces(marker=dict(size=4, sizemode='diameter', sizeref=1, sizemin=4))
    else:
        fig = px.scatter(df, x="x", y="y", color="label", width=700, height=900)

    st.plotly_chart(fig, use_container_width=True, key=f"plot-{uuid.uuid4()}")


def show_embedding_projector(
    model: torch.nn.Module,
    images: List[Image.Image],
    labels: List[str],
    device: str = "cuda"
):
    if not images:
        st.warning("Keine Bilder übergeben.")
        return

    # Layer-Auswahl
    st.subheader("Layer-Auswahl")
    possible_layers = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]
    layer_name = st.selectbox("Layer für Aktivierung", possible_layers, index=5)  # default: layer4

    st.info(f"Sammle Aktivierungen aus Layer **{layer_name}** …")
    vecs = _collect_activations(model, images, layer_name, device=device)

    st.subheader("Dimensionale Reduktion")
    method = st.selectbox("Methode", ["umap", "tsne", "pca", "mds"])

    dim = st.sidebar.selectbox("Ziel-Dimension", [2, 3], index=0)
    st.info(f"Reduziere von {vecs.shape[1]} → {dim} Dimensionen mit {method.upper()} …")

    params = {}
    if method == "umap":
        params["n_neighbors"] = st.slider("n_neighbors", 5, 50, 15)
        params["min_dist"] = st.slider("min_dist", 0.0, 1.0, 0.1)
    elif method == "tsne":
        params["perplexity"] = st.slider("Perplexity", 5, 100, 30)
        params["max_iter"] = st.slider("Iterationen", 250, 2000, 1000)
    elif method == "mds":
        params["n_init"] = st.slider("Anzahl Initialisierungen", 1, 10, 4)
        params["max_iter"] = st.slider("Maximale Iterationen", 100, 1000, 300)

    bar = st.progress(0.0, text=f"Reduziere mit {method.upper()} …")
    emb = _reduce(vecs, method=method, dim=dim, params=params)
    bar.progress(1.0)
    bar.empty()

    _plot(emb, labels, dim)
