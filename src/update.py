from typing import Optional, List, Tuple
import streamlit as st
from PIL import Image
from torchvision.datasets import MNIST, CIFAR10
import random
import torchvision.transforms as T


def upload_image() -> Optional[Image.Image]:
    file = st.sidebar.file_uploader("Eingabebild hochladen", ["png", "jpg", "jpeg"])
    if file:
        img = Image.open(file).convert("RGB")
        st.sidebar.image(img, caption="Vorschau", use_container_width=True)
        return img
    return None

_DATASETS = {
    "None / Einzelbild": None,
    "MNIST Demo (1k)": "mnist1k",
    "CIFAR10 Demo (1k)": "cifar10_1k",
}

def dataset_select()->str:
    return st.sidebar.selectbox("Demo‑Datensatz", list(_DATASETS.keys()))

@st.cache_resource(show_spinner="Lade MNIST (1k) …")
def load_demo_dataset(name: str) -> Tuple[List[Image.Image], List[str]]:
    if name == "mnist1k":
        ds = MNIST(root="./data", train=True, download=True)
        idx = random.sample(range(len(ds)), 1000)
        imgs, labels = [], []
        for i in idx:
            pil_img, label = ds[i]              # MNIST liefert bereits PIL‑Image
            pil_img = pil_img.convert("RGB").resize((224, 224))
            imgs.append(pil_img)
            labels.append(f"Ziffer {label}")
        return imgs, labels

    elif name == "cifar10_1k":
        CIFAR10_CLASSES = [
            "Flugzeug", "Auto", "Vogel", "Katze", "Hirsch",
            "Hund", "Frosch", "Pferd", "Schiff", "LKW"
        ]
        ds = CIFAR10(root="./data", train=True, download=True)
        transform = T.Resize((224, 224))
        idx = random.sample(range(len(ds)), 1000)
        imgs, labels = [], []
        for i in idx:
            img, label = ds[i]
            img = transform(img)
            imgs.append(img)
            labels.append(CIFAR10_CLASSES[label])  # ← Namen statt Zahl
        return imgs, labels

    raise ValueError(name)


def show_placeholder(method: str):
    st.info(f"{method} wird hier angezeigt, sobald implementiert.")
