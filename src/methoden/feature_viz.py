import torch
import matplotlib.pyplot as plt
import streamlit as st
import torchvision.transforms as transforms
import cv2
import numpy as np

activation = {}

def register_activation_hook(model, model_type):
    # Reset vorherige Hooks
    for handle in getattr(model, "hook_handles", []):
        handle.remove()
    model.hook_handles = []

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    if model_type.lower() == "resnet50":
        layer = model.layer4
    elif model_type.lower() == "vgg16":
        layer = model.features
    elif model_type.lower() == "mobilenet":
        layer = model.features
    else:
        raise ValueError("Unbekannter Modelltyp")

    handle = layer.register_forward_hook(get_activation("features"))
    model.hook_handles.append(handle)


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)
    tensor.requires_grad_()  # ← Wichtig für Hooks
    return tensor


def show_feature_maps(model, image, model_type='resnet50'):
    register_activation_hook(model, model_type)
    input_tensor = preprocess_image(image)

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    _ = model(input_tensor)

    if 'features' not in activation:
        st.warning("Keine Features gefunden.")
        return

    act = activation['features'][0]
    C, H, W = act.shape

    channel_score = act.abs().mean(dim=(1, 2))

    top_k = min(16, C)
    top_indices = torch.topk(channel_score, k=top_k, largest=True).indices

    rows = cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for i, ch_idx in enumerate(top_indices):
        ax = axes[i // cols, i % cols]
        fmap_np = act[ch_idx].cpu().numpy()
        ax.imshow(fmap_np, cmap='viridis')
        ax.set_title(f"ch {ch_idx.item()}")
        ax.axis('off')

    plt.tight_layout()
    st.pyplot(fig)


def show_feature_overlay(model, image, model_type='resnet50'):
    register_activation_hook(model, model_type)
    input_tensor = preprocess_image(image)

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    _ = model(input_tensor)

    if 'features' not in activation:
        st.warning("Keine Features gefunden.")
        return

    act = activation['features'][0]
    fmap = act.mean(0).cpu().numpy()
    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())
    fmap_resized = cv2.resize(fmap, image.size)

    heatmap = np.uint8(255 * fmap_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    orig_np = np.array(image.convert("RGB"))
    overlay = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)

    st.image(overlay, caption="Feature Map Overlay")