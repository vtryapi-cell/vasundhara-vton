```python
import os
import base64

from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set.\n"
        "Run:\n"
        "set GEMINI_API_KEY=YOUR_API_KEY"
    )


PERSON_IMAGE = "person.jpg"
SAREE_IMAGE = "saree.jpg"

OUTPUT_IMAGE = "gemini_saree_result.png"

MODEL = "gemini-3.1-flash-image"


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(PERSON_IMAGE):
    raise FileNotFoundError(
        f"Person image not found: {PERSON_IMAGE}"
    )

if not os.path.exists(SAREE_IMAGE):
    raise FileNotFoundError(
        f"Saree image not found: {SAREE_IMAGE}"
    )


# ============================================================
# READ IMAGES
# ============================================================

with open(PERSON_IMAGE, "rb") as f:
    person_bytes = f.read()

with open(SAREE_IMAGE, "rb") as f:
    saree_bytes = f.read()


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# TRY-ON PROMPT
# ============================================================

prompt = """
Create a highly realistic Indian fashion virtual try-on image.

IMAGE 1 is the PERSON.
IMAGE 2 is the SAREE PRODUCT.

TASK:

Take the exact saree design, colors, border, pattern,
fabric appearance and decorative details from IMAGE 2.

Make the woman/person from IMAGE 1 wear that saree.

The result must look like a real photograph of the SAME PERSON
wearing the provided saree.

IMPORTANT:

- Preserve the person's face and identity.
- Preserve the person's hairstyle as much as possible.
- Preserve natural body proportions.
- Preserve the original person's skin tone.
- Preserve the original person's pose as much as practical.
- Replace the existing clothing with the saree.
- Create a realistic Indian saree drape.
- Create clearly visible saree pleats.
- Create a realistic pallu draped naturally over the shoulder.
- Preserve the exact saree colors and printed/embroidered design.
- Preserve the saree border.
- Make the saree fabric follow the person's body naturally.
- Add realistic fabric folds and shadows.
- Make the blouse look appropriate to the saree.
- Make the blouse visually coordinated with the saree.
- Do not turn the saree into pants, trousers, leggings, or a western dress.
- Do not change the saree into a generic outfit.
- Do not invent a completely different saree.
- Do not change the person's face.
- Do not add extra people.
- Do not add jewelry unless it naturally belongs to the provided outfit.

The final result should look like a professional Indian fashion
e-commerce photograph showing the SAME WOMAN wearing the
EXACT PROVIDED SAREE.

Photorealistic result.
Natural lighting.
Realistic fabric texture.
Realistic shadows.
High detail.
Full-body composition.
"""


# ============================================================
# GEMINI REQUEST
# ============================================================

print("========================================")
print("GEMINI SAREE TRY-ON TEST")
print("========================================")

print("Person:", PERSON_IMAGE)
print("Saree :", SAREE_IMAGE)
print("Model :", MODEL)
print()
print("Sending images to Gemini...")
print()


interaction = client.interactions.create(
    model=MODEL,

    input=[
        {
            "type": "image",
            "data": base64.b64encode(
                person_bytes
            ).decode("utf-8"),
            "mime_type": "image/jpeg",
        },

        {
            "type": "image",
            "data": base64.b64encode(
                saree_bytes
            ).decode("utf-8"),
            "mime_type": "image/jpeg",
        },

        {
            "type": "text",
            "text": prompt,
        },
    ],

    response_format={
        "type": "image",
        "mime_type": "image/png",
        "aspect_ratio": "3:4",
        "image_size": "1K",
    },
)


# ============================================================
# SAVE RESULT
# ============================================================

if interaction.output_image:

    image_data = interaction.output_image.data

    with open(
        OUTPUT_IMAGE,
        "wb"
    ) as f:

        f.write(
            base64.b64decode(
                image_data
            )
        )

    print("========================================")
    print("SUCCESS")
    print("========================================")
    print()
    print(
        "Result saved as:",
        os.path.abspath(OUTPUT_IMAGE)
    )

else:

    print("========================================")
    print("NO IMAGE RETURNED")
    print("========================================")

    print(
        "Gemini response:"
    )

    print(
        interaction
    )
```
