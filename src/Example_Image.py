from PIL import Image

def load_example_image():
    return Image.open("resources/beagle.jpg").convert("RGB")
