import streamlit as st
import copy

from .methoden import gradcam ,feature_viz, embedding_proj
from . import update
from .Example_Image import load_example_image  # Funktion zum Laden des Beispielbildes


_METHODS = ["Grad-CAM", "Saliency Map", "LIME", "Feature Viz", "Embedding Projector"]



def render_main_layout(model_name:str, model_obj):
    st.header(f"Modell: {model_name}")

    prev_choice = st.session_state.get("prev_dataset", None)
    choice = update.dataset_select()

    if choice != prev_choice:
        if choice != "None / Einzelbild":
            st.session_state["selected_tab"] = "Embedding Projector"
        else:
            st.session_state["selected_tab"] = "Grad-CAM"
    st.session_state["prev_dataset"] = choice

    if choice == "None / Einzelbild":
        image = update.upload_image()
        images, labels = ([image], ["Input"]) if image else ([], [])
    else:
        images, labels = update.load_demo_dataset(update._DATASETS[choice])
        st.sidebar.success(f"{len(images)} Bilder geladen")

    if not images:
        st.info("Bitte Bild hochladen oder Datensatz wählen.")
        return

    #active_tab_index = _METHODS.index(st.session_state.get("selected_tab", "Grad-CAM"))
    #tabs = st.tabs(_METHODS, index=active_tab_index)

    for tab, method in zip(st.tabs(_METHODS), _METHODS):
        with tab:
            st.subheader(method)
            if choice != "None / Einzelbild" and method != "Embedding Projector":
                st.info("Für diesen Datensatz ist nur der Embedding Projector verfügbar.")
                continue

            if method == "Grad-CAM":
                gradcam.show_gradcam(copy.deepcopy(model_obj), images[0])
            elif method == "Feature Viz":
                feature_viz.show_feature_maps(model_obj, images[0], model_name)
                st.divider()
                feature_viz.show_feature_overlay(model_obj, images[0], model_name)
            elif method == "Embedding Projector":
                embedding_proj.show_embedding_projector(model_obj, images=images, labels=labels, device="cuda")
            else:
                update.show_placeholder(method)

            