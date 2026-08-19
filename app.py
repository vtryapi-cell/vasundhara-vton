```python
import os
import base64
import time
import requests

from flask import Flask, jsonify, request, render_template
from openai import OpenAI

app = Flask(__name__)

# ============================================================
# API CONFIGURATION
# ============================================================

FASHN_API_KEY = os.environ.get("FASHN_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

FASHN_URL = "https://api.fashn.ai/v1"

openai_client = None

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

print("🔥 VASUNDHARA VTON APP LOADED")
print("APP FILE:", __file__)
print("FASHN API KEY PRESENT:", bool(FASHN_API_KEY))
print("OPENAI API KEY PRESENT:", bool(OPENAI_API_KEY))


# ============================================================
# IMAGE HELPER
# ============================================================

def image_to_data_uri(file):

    data = file.read()

    if not data:
        raise ValueError("Uploaded image is empty")

    content_type = file.content_type or "image/jpeg"

    encoded = base64.b64encode(data).decode("utf-8")

    return f"data:{content_type};base64,{encoded}"


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    try:
        return render_template("index.html")

    except Exception:

        return """
        <h1>VASUNDHARA VTON IS LIVE</h1>
        <p>Website is running.</p>
        """


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return jsonify({
        "ok": True,
        "message": "VASUNDHARA VTON SERVER IS RUNNING",
        "fashn_key_present": bool(FASHN_API_KEY),
        "openai_key_present": bool(OPENAI_API_KEY)
    })


# ============================================================
# DEBUG
# ============================================================

@app.get("/debug")
def debug():

    return jsonify({
        "status": "OK",
        "app_file": __file__,
        "fashn_key_present": bool(FASHN_API_KEY),
        "openai_key_present": bool(OPENAI_API_KEY),
        "routes": [
            str(rule)
            for rule in app.url_map.iter_rules()
        ]
    })


# ============================================================
# FASHN TRY-ON
# ============================================================

@app.post("/api/tryon")
def tryon():

    if not FASHN_API_KEY:

        return jsonify({
            "success": False,
            "error": "FASHN_API_KEY is not configured in Cloud Run."
        }), 500

    if "model_image" not in request.files:

        return jsonify({
            "success": False,
            "error": "model_image is missing."
        }), 400

    if "garment_image" not in request.files:

        return jsonify({
            "success": False,
            "error": "garment_image is missing."
        }), 400

    try:

        model_file = request.files["model_image"]
        garment_file = request.files["garment_image"]

        model_image = image_to_data_uri(model_file)
        garment_image = image_to_data_uri(garment_file)

        payload = {
            "model_name": "tryon-v1.6",

            "inputs": {

                "model_image": model_image,

                "garment_image": garment_image,

                "category": "auto",

                "garment_photo_type": "auto",

                "mode": "balanced",

                "num_samples": 1,

                "output_format": "jpeg"
            }
        }

        headers = {

            "Authorization": f"Bearer {FASHN_API_KEY}",

            "Content-Type": "application/json"
        }

        print("🚀 Sending try-on request to FASHN...")

        response = requests.post(

            f"{FASHN_URL}/run",

            json=payload,

            headers=headers,

            timeout=60
        )

        print(
            "FASHN RUN STATUS:",
            response.status_code
        )

        print(
            "FASHN RESPONSE:",
            response.text[:1000]
        )

        if not response.ok:

            return jsonify({

                "success": False,

                "error": "FASHN API request failed.",

                "details": response.text

            }), 502

        run_data = response.json()

        prediction_id = run_data.get("id")

        if not prediction_id:

            return jsonify({

                "success": False,

                "error": "FASHN did not return a prediction ID.",

                "response": run_data

            }), 502

        # ----------------------------------------------------
        # POLL FASHN
        # ----------------------------------------------------

        for _ in range(60):

            time.sleep(2)

            status_response = requests.get(

                f"{FASHN_URL}/status/{prediction_id}",

                headers={

                    "Authorization":
                    f"Bearer {FASHN_API_KEY}"

                },

                timeout=30
            )

            status_data = status_response.json()

            status = status_data.get("status")

            print(
                "FASHN STATUS:",
                status
            )

            if status == "completed":

                output = status_data.get(
                    "output",
                    []
                )

                if not output:

                    return jsonify({

                        "success": False,

                        "error":
                        "FASHN completed but returned no output."

                    }), 502

                return jsonify({

                    "success": True,

                    "image_url": output[0],

                    "prediction_id":
                    prediction_id,

                    "provider":
                    "fashn"
                })

            if status in [

                "failed",

                "canceled",

                "cancelled",

                "error"

            ]:

                return jsonify({

                    "success": False,

                    "error":
                    "FASHN try-on failed.",

                    "details":
                    status_data.get("error"),

                    "prediction_id":
                    prediction_id

                }), 502

        return jsonify({

            "success": False,

            "error":
            "FASHN prediction timed out.",

            "prediction_id":
            prediction_id

        }), 504

    except Exception as e:

        print(
            "❌ FASHN TRY-ON ERROR:",
            repr(e)
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# OPENAI SAREE TRY-ON
# ============================================================

@app.post("/api/openai-tryon")
def openai_tryon():

    print("🤖 OPENAI SAREE TRY-ON REQUEST RECEIVED")

    if not OPENAI_API_KEY or not openai_client:

        return jsonify({

            "success": False,

            "error":
            "OPENAI_API_KEY is not configured in Cloud Run."

        }), 500

    if "model_image" not in request.files:

        return jsonify({

            "success": False,

            "error":
            "model_image is missing."

        }), 400

    if "garment_image" not in request.files:

        return jsonify({

            "success": False,

            "error":
            "garment_image is missing."

        }), 400

    try:

        model_file = request.files[
            "model_image"
        ]

        garment_file = request.files[
            "garment_image"
        ]

        model_data = model_file.read()

        garment_data = garment_file.read()

        if not model_data:

            raise ValueError(
                "Model image is empty."
            )

        if not garment_data:

            raise ValueError(
                "Garment image is empty."
            )

        model_type = (
            model_file.content_type
            or "image/jpeg"
        )

        garment_type = (
            garment_file.content_type
            or "image/jpeg"
        )

        model_b64 = base64.b64encode(
            model_data
        ).decode("utf-8")

        garment_b64 = base64.b64encode(
            garment_data
        ).decode("utf-8")

        model_uri = (
            f"data:{model_type};base64,{model_b64}"
        )

        garment_uri = (
            f"data:{garment_type};base64,{garment_b64}"
        )

        # ----------------------------------------------------
        # SAREE PROMPT
        # ----------------------------------------------------

        prompt = """
Create a photorealistic virtual try-on image.

IMAGE 1 is the person/model.
IMAGE 2 is the saree product photograph.

Dress the person in IMAGE 1 with the EXACT saree shown
in IMAGE 2.

IMPORTANT:

This is an INDIAN SAREE.

Do NOT turn the saree into:

- a dress
- a gown
- a skirt
- a western one-piece
- a jumpsuit
- a lehenga

Instead, create a realistic traditional Indian saree drape.

The saree should wrap naturally around the lower body
and waist and the pallu should be draped naturally over
one shoulder.

Preserve the exact visual identity of the saree product:

- green color
- orange color
- saree border
- woven pattern
- pallu
- fabric appearance
- overall design

Use the product image as the garment reference.

Keep the model's:

- face
- identity
- skin tone
- body proportions
- hairstyle
- pose

as close as possible to IMAGE 1.

Do not change the person's identity.

Keep the background and lighting close to IMAGE 1.

The final image should look like a professional
Indian fashion e-commerce photograph.

The result must clearly look like the SAME PERSON
wearing the SAME SAREE shown in the product image.

Make the saree draping physically realistic and natural.
"""


        print(
            "🚀 Sending images to OpenAI..."
        )

        # ----------------------------------------------------
        # OPENAI IMAGE GENERATION
        # ----------------------------------------------------

        result = openai_client.images.edit(

            model="gpt-image-2",

            image=[
                model_uri,
                garment_uri
            ],

            prompt=prompt,

            size="1024x1536",

            quality="medium"
        )

        print(
            "✅ OpenAI response received"
        )

        if not result.data:

            return jsonify({

                "success": False,

                "error":
                "OpenAI returned no image."

            }), 502

        image_b64 = result.data[0].b64_json

        if not image_b64:

            return jsonify({

                "success": False,

                "error":
                "OpenAI returned an empty image."

            }), 502

        image_url = (
            "data:image/png;base64,"
            + image_b64
        )

        print(
            "✅ OPENAI SAREE RESULT READY"
        )

        return jsonify({

            "success": True,

            "image_url": image_url,

            "provider": "openai",

            "model": "gpt-image-2"

        })

    except Exception as e:

        print(
            "❌ OPENAI TRY-ON ERROR:",
            repr(e)
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# 404 HANDLER
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "error": "Flask 404",

        "requested_path":
        request.path

    }), 404


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "8080"
        )
    )

    app.run(

        host="0.0.0.0",

        port=port
    )
```
