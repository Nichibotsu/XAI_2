import torch
import matplotlib.pyplot as plt
import streamlit as st
import torchvision.transforms as transforms

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


    # Kein torch.no_grad() verwenden, damit Hook funktioniert!
    _ = model(input_tensor)

    if 'features' not in activation:
        st.warning("Keine Features gefunden.")
        return

    act = activation['features'][0]
    num_channels = min(16, act.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(num_channels):
        ax = axes[i // 4, i % 4]
        ax.imshow(act[i].cpu(), cmap='viridis')
        ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)

import cv2
import numpy as np

def show_feature_overlay(model, image, model_type='resnet50'):
    register_activation_hook(model, model_type)
    input_tensor = preprocess_image(image)

    _ = model(input_tensor)  # Forward-Pass, Hook füllt `activation`

    if 'features' not in activation:
        st.warning("Keine Features gefunden.")
        return

    act = activation['features'][0]  # [C, H, W]
    fmap = act.mean(0).cpu().numpy()  # Mittel über alle Kanäle
    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())  # Normieren
    fmap_resized = cv2.resize(fmap, image.size)  # (width, height)

    heatmap = np.uint8(255 * fmap_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    orig_np = np.array(image.convert("RGB"))  # Originalbild als NumPy
    overlay = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)

    st.image(overlay, caption="Feature Map Overlay")
