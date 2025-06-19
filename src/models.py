from typing import Dict, Callable, Any, List
import streamlit as st
from torchvision.models import resnet50, ResNet50_Weights

@st.cache_resource(show_spinner="Lade ResNet50 …")
def load_resnet50():

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def get_imagenet_labels() -> List[str]:
    try:
        return ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
    except Exception:
        return [f"Klasse {i}" for i in range(1000)]


def _not_implemented(name: str) -> Any:
    st.warning(f"Ladefunktion für {name} ist noch nicht implementiert.")
    return None


def load_vgg16():
    return _not_implemented("VGG16")


def load_mobilenet():
    return _not_implemented("MobileNet")


MODELS: Dict[str, Callable[[], Any]] = {
    "ResNet50": load_resnet50,
    "VGG16": load_vgg16,
    "MobileNet": load_mobilenet,
}