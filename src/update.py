from typing import Optional
import streamlit as st
from PIL import Image


def upload_image() -> Optional[Image.Image]:
    file = st.sidebar.file_uploader("Eingabebild hochladen", ["png", "jpg", "jpeg"])
    if file:
        img = Image.open(file).convert("RGB")
        st.sidebar.image(img, caption="Vorschau", use_container_width=True)
        return img
    return None


def show_placeholder(method: str):
    st.info(f"{method} wird hier angezeigt, sobald implementiert.")
