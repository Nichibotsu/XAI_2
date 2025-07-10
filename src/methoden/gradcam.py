from typing import List, Optional
import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchcam.methods import GradCAM
from ..models import get_imagenet_labels

def _preprocess(img: Image.Image) -> torch.Tensor:
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

def _overlay(base: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
    cam = (cam * 255).astype(np.uint8)
    heat = Image.fromarray(cam, mode="L").resize(base.size, Image.BILINEAR).convert("RGBA")
    r, *_ = heat.split()
    red_heat = Image.merge("RGBA", (r, Image.new("L", r.size), Image.new("L", r.size), r))
    return Image.blend(base.convert("RGBA"), red_heat, alpha).convert("RGB")

def _get_possible_cam_layers(model: torch.nn.Module) -> List[str]:
    name = model.__class__.__name__.lower()
    if "resnet" in name:
        return ["conv1", "layer1", "layer2", "layer3", "layer4"]
    elif "vgg" in name:
        return [f"features.{i}" for i in [5, 10, 17, 24, 28]]
    elif "mobilenet" in name:
        return [f"features.{i}" for i in [0, 3, 6, 13, 17]]
    else:
        return []

def _get_cam_extractor(model: torch.nn.Module, layer: str):
    return GradCAM(model, target_layer=layer)


def _predict(model: torch.nn.Module, tensor: torch.Tensor, labels: List[str]):
    logits = model(tensor)
    probs = F.softmax(logits, dim=1)
    top_prob, top_idx = probs[0].topk(1)
    label = labels[top_idx.item()] if top_idx.item() < len(labels) else str(top_idx.item())
    return label, float(top_prob.item()), logits


def _compute_cam(extractor: GradCAM, logits: torch.Tensor) -> np.ndarray:
    cls = int(logits.argmax(1))
    cam_tensor = extractor(cls, logits)[0].detach().cpu()
    if cam_tensor.ndim == 3:
        cam_tensor = cam_tensor.mean(0)
    cam = cam_tensor.numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)


def show_gradcam(model: Optional[torch.nn.Module], img: Image.Image):
    if model is None:
        st.warning("Kein Modell geladen – Grad‑CAM nicht möglich.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # 👉 Sidebar-Layer-Auswahl
    st.sidebar.subheader("Grad‑CAM Einstellungen")
    possible_layers = _get_possible_cam_layers(model)
    if not possible_layers:
        st.error("Für dieses Modell sind keine Grad‑CAM Layer definiert.")
        return

    selected_layer = st.sidebar.selectbox("Target-Layer für Grad‑CAM", possible_layers, index=len(possible_layers)-1)
    extractor = _get_cam_extractor(model, selected_layer)

    tensor = _preprocess(img).to(device)
    labels = get_imagenet_labels()

    with st.spinner("Inference & Grad‑CAM …"):
        try:
            label, prob, logits = _predict(model, tensor, labels)
            cam = _compute_cam(extractor, logits)
            overlay = _overlay(img, cam)
        except Exception as e:
            st.error(f"Grad‑CAM Fehler: {e}")
            return

    st.success(f"**Vorhersage:** {label}  •  Konfidenz: {prob:.1%}")
    col1, col2 = st.columns(2)
    col1.image(img, caption="Original", use_container_width=True)
    col2.image(overlay, caption=f"Grad‑CAM ({selected_layer})", use_container_width=True)