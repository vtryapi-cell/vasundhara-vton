import os
import io
import time
import base64
import requests

from flask import Flask, jsonify, request, render_template
from openai import OpenAI

app = Flask(__name__)

# ============================================================
# API KEYS
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
# HOME
# ============================================================

@app.get("/")
def home():
    try:
        return render_template("index.html")
    except Exception as e:
        print("HOME ERROR:", repr(e))
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

        model_data = model_file.read()
        garment_data = garment_file.read()

        if not model_data:
            raise ValueError("Model image is empty.")

        if not garment_data:
            raise ValueError("Garment image is empty.")

        model_type = model_file.content_type or "image/jpeg"
        garment_type = garment_file.content_type or "image/jpeg"

        model_image = (
            f"data:{model_type};base64,"
            + base64.b64encode(model_data).decode("utf-8")
        )

        garment_image = (
            f"data:{garment_type};base64,"
            + base64.b64encode(garment_data).decode("utf-8")
        )

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

        print("🚀 Sending request to FASHN...")

        response = requests.post(
            f"{FASHN_URL}/run",
            json=payload,
            headers=headers,
            timeout=60
        )

        print("FASHN STATUS:", response.status_code)
        print("FASHN RESPONSE:", response.text[:1000])

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

        for _ in range(60):

            time.sleep(2)

            status_response = requests.get(
                f"{FASHN_URL}/status/{prediction_id}",
                headers={
                    "Authorization": f"Bearer {FASHN_API_KEY}"
                },
                timeout=30
            )

            status_data = status_response.json()
            status = status_data.get("status")

            print("FASHN STATUS:", status)

            if status == "completed":

                output = status_data.get("output", [])

                if not output:
                    return jsonify({
                        "success": False,
                        "error": "FASHN completed but returned no output."
                    }), 502

                return jsonify({
                    "success": True,
                    "image_url": output[0],
                    "prediction_id": prediction_id,
                    "provider": "fashn"
                })

            if status in [
                "failed",
                "canceled",
                "cancelled",
                "error"
            ]:

                return jsonify({
                    "success": False,
                    "error": "FASHN try-on failed.",
                    "details": status_data.get("error"),
                    "prediction_id": prediction_id
                }), 502

        return jsonify({
            "success": False,
            "error": "FASHN prediction timed out.",
            "prediction_id": prediction_id
        }), 504

    except Exception as e:

        print("❌ FASHN ERROR:", repr(e))

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
            "error": "OPENAI_API_KEY is not configured in Cloud Run."
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

        model_data = model_file.read()
        garment_data = garment_file.read()

        if not model_data:
            raise ValueError("Model image is empty.")

        if not garment_data:
            raise ValueError("Garment image is empty.")

        model_buffer = io.BytesIO(model_data)
        model_buffer.name = model_file.filename or "model.jpg"

        garment_buffer = io.BytesIO(garment_data)
        garment_buffer.name = garment_file.filename or "saree.jpg"

        prompt = """
Create a photorealistic Indian fashion virtual try-on.

IMAGE 1 is the person/model.
IMAGE 2 is the saree product photograph.

Dress the person in IMAGE 1 with the SAME saree shown in IMAGE 2.

This MUST be a traditional Indian saree.

Do NOT turn the saree into:
- a dress
- a gown
- a skirt
- a western one-piece
- a jumpsuit
- a lehenga

Create realistic traditional Indian saree draping.

The saree must:
- wrap naturally around the waist
- cover the lower body naturally
- have realistic saree pleats
- have a natural pallu
- place the pallu naturally over one shoulder
- look physically believable

Preserve the actual product design from IMAGE 2.

Preserve:
- green color
- orange color
- orange border
- woven patterns
- pallu design
- fabric appearance
- overall saree design

Do not replace the product with a generic saree.

Keep the person from IMAGE 1 as the same person.

Preserve:
- face
- facial features
- identity
- hairstyle
- skin tone
- body proportions
- pose

Keep the background and lighting close to IMAGE 1.

The final result should look like a professional
Indian fashion e-commerce photograph.

The result must clearly show the same person
wearing the same saree product.

Make the saree draping realistic, elegant and natural.
"""

        print("🚀 Sending model + saree to OpenAI...")

        result = openai_client.images.edit(
            model="gpt-image-2",
            image=[
                model_buffer,
                garment_buffer
            ],
            prompt=prompt,
            size="1024x1536",
            quality="medium"
        )

        print("✅ OpenAI response received")

        if not result.data:
            return jsonify({
                "success": False,
                "error": "OpenAI returned no image."
            }), 502

        image_b64 = result.data[0].b64_json

        if not image_b64:
            return jsonify({
                "success": False,
                "error": "OpenAI returned an empty image."
            }), 502

        image_url = (
            "data:image/png;base64,"
            + image_b64
        )

        print("✅ OPENAI SAREE RESULT READY")

        return jsonify({
            "success": True,
            "image_url": image_url,
            "provider": "openai",
            "model": "gpt-image-2"
        })

    except Exception as e:

        print("❌ OPENAI TRY-ON ERROR:", repr(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# 404
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({
        "error": "Flask 404",
        "requested_path": request.path
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
