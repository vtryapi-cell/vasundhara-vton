from PIL import Image


# =========================================================
# VASUNDHARA VTON V7 - PREPROCESSING
# =========================================================

MAX_IMAGE_SIZE = 1536


def load_image(image):
    """
    Convert input to RGB PIL image.
    """

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    raise TypeError("Input must be a PIL Image")


def resize_image(image, max_size=MAX_IMAGE_SIZE):
    """
    Resize while preserving aspect ratio.
    """

    image = load_image(image)

    width, height = image.size

    if max(width, height) <= max_size:
        return image

    scale = max_size / max(width, height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )


def prepare_person(image):
    """
    Prepare the person image.

    Face and body information are preserved.
    """

    image = load_image(image)
    image = resize_image(image)

    return image


def prepare_garment(image):
    """
    Prepare the garment/saree image.

    The original garment colors and patterns are preserved.
    """

    image = load_image(image)
    image = resize_image(image)

    return image


def prepare_inputs(person_image, garment_image):
    """
    Prepare person and garment for VASUNDHARA VTON.
    """

    person = prepare_person(person_image)
    garment = prepare_garment(garment_image)

    return {
        "person": person,
        "garment": garment,
    }
