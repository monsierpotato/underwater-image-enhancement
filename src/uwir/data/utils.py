from PIL import Image


def is_image_file(filename):
    return any(
        filename.endswith(ext) for ext in [".png", ".jpg", ".bmp", ".JPG", ".jpeg", ".PNG", ".JPEG"]
    )


def load_img(filepath):
    img = Image.open(filepath).convert("RGB")
    return img
