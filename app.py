import streamlit as st
from src.models import MODELS
from src.layout import render_main_layout

def main():
    st.set_page_config(page_title="CNN Explainability Dashboard", layout="wide")
    st.title("Explainability Dashboard für CNN-Modelle")

    # Seitenleiste – Modellwahl
    model_key = st.sidebar.selectbox("Wähle ein Modell", list(MODELS.keys()))

    # Lazy‑Loading des Modells (Platzhalter)
    model = MODELS[model_key]()

    # Hauptlayout (Tabs für Explainability‑Methoden)
    render_main_layout(model_key, model)


if __name__ == "__main__":
    main()