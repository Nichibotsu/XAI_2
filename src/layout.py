import streamlit as st

from .methoden import gradcam
from . import update

_METHODS = ["Grad-CAM", "Saliency Map", "LIME", "Feature Viz"]


def render_main_layout(model_name: str, model_obj):
    st.header(f"Modell: {model_name}")
    image = update.upload_image()
    if image is None:
        st.info("Bitte lade in der Seitenleiste ein Bild hoch.")
        return

    tabs = st.tabs(_METHODS)
    for tab, method in zip(tabs, _METHODS):
        with tab:
            st.subheader(method)
            if method == "Grad-CAM":
                gradcam.show_gradcam(model_obj, image)
            else:
                update.show_placeholder(method)