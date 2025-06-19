import streamlit as st
import copy

from .methoden import gradcam ,feature_viz
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
                grad_model_obj=copy.deepcopy(model_obj)
                gradcam.show_gradcam(grad_model_obj, image)
            elif method == "Feature Viz":
                feature_viz.show_feature_maps(model_obj, image, model_name)
            else:
                update.show_placeholder(method)
            