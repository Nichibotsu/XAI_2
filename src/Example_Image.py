from PIL import Image

def load_example_image():
    return Image.open("Beispiel-Bild/Katze.jpg").convert("RGB")
