from .model import VasundharaVTON


_ENGINE = None


def get_engine():
    global _ENGINE

    if _ENGINE is None:
        _ENGINE = VasundharaVTON()

    return _ENGINE


def generate_tryon(
    person_image,
    garment_image,
    category="one-pieces",
):
    engine = get_engine()

    return engine.generate(
        person_image=person_image,
        garment_image=garment_image,
        category=category,
    )
