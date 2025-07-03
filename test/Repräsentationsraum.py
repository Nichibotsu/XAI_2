import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import umap
import plotly.express as px
from PIL import Image
import os

st.set_page_config(layout="wide")
st.title("🔍 Oxford-Pets Feature Visualizer (2D & 3D)")

# Sidebar
model_name = st.sidebar.selectbox("Wähle Modell", ["resnet50", "vgg16", "mobilenet"])
reduction_method = st.sidebar.selectbox("Dimensionsreduktion", ["tsne", "umap"])
plot_mode = st.sidebar.selectbox("Darstellung", ["2D", "3D"])
data_dir = st.sidebar.text_input("Pfad zu Bilddaten (ImageFolder)", "./data/mini_subset")

image_size = 224
batch_size = 32

# Datenprüfung
if not os.path.exists(data_dir):
    st.warning("❗ Bitte gib einen gültigen Datenpfad an – z. B. './data/mini_subset'")
    st.stop()

# Transformation & Dataset
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
raw_transform = transforms.Resize((64, 64))

dataset = ImageFolder(data_dir, transform=transform)
raw_dataset = ImageFolder(data_dir, transform=raw_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
class_names = dataset.classes

# Modellfunktion
@st.cache_resource
def get_model(name):
    if name == "resnet50":
        m = models.resnet50(pretrained=True)
        return torch.nn.Sequential(*(list(m.children())[:-1])).eval()
    elif name == "vgg16":
        m = models.vgg16(pretrained=True)
        return torch.nn.Sequential(*(list(m.features.children()))).eval()
    elif name == "mobilenet":
        m = models.mobilenet_v2(pretrained=True)
        return m.features.eval()

# Feature-Extraktion
@st.cache_data
def extract_features(_model, _loader):
    feats, labels = [], []
    with torch.no_grad():
        for x, y in _loader:
            f = _model(x)
            if f.dim() == 4:
                f = torch.nn.functional.adaptive_avg_pool2d(f, (1, 1))
            f = f.view(f.size(0), -1)
            feats.append(f)
            labels.extend(y.numpy())
    return torch.cat(feats).numpy(), labels

# Dimensionsreduktion
@st.cache_data
def reduce_features(features, method, n_components=3):
    if method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
    else:
        reducer = umap.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(features)

# Modell laden
model = get_model(model_name)
features, labels = extract_features(model, dataloader)
n_dim = 3 if plot_mode == "3D" else 2
embedding = reduce_features(features, reduction_method, n_components=n_dim)
label_names = [class_names[l] for l in labels]

# Visualisierung
st.subheader(f"📊 {plot_mode} Feature-Space Visualisierung")

if plot_mode == "2D":
    fig = px.scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        color=label_names,
        hover_data={"Label": label_names},
        labels={"x": "Dimension 1", "y": "Dimension 2"},
        title=f"{model_name.upper()} Features mit {reduction_method.upper()} (2D)"
    )
    fig.update_traces(marker=dict(size=7, opacity=0.7, line=dict(width=0.5, color='DarkSlateGrey')))
else:
    fig = px.scatter_3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        color=label_names,
        hover_data={"Label": label_names},
        labels={"x": "Dim 1", "y": "Dim 2", "z": "Dim 3"},
        title=f"{model_name.upper()} Features mit {reduction_method.upper()} (3D)"
    )
    fig.update_traces(marker=dict(size=5, opacity=0.75, line=dict(width=0.5, color='DarkSlateGrey')))

st.plotly_chart(fig, use_container_width=True)

# Bildvorschau
st.subheader("🔍 Bildvorschau")
idx = st.slider("Beispielindex", 0, len(raw_dataset) - 1, 0)
img, lbl = raw_dataset[idx]
st.image(img, caption=f"Klasse: {class_names[lbl]}", width=256)
