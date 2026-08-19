import os
import base64
import time
import requests

from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

FASHN_API_KEY = os.environ.get("FASHN_API_KEY")
FASHN_URL = "https://api.fashn.ai/v1"


print("🔥 VASUNDHARA VTON APP LOADED")
print("APP FILE:", __file__)
print("FASHN API KEY PRESENT:", bool(FASHN_API_KEY))


def image_to_data_uri(file):
    """Convert uploaded image to a FASHN-compatible data URI."""

    data = file.read()

    if not data:
        raise ValueError("Uploaded image is empty")

    content_type = file.content_type or "image/jpeg"

    encoded = base64.b64encode(data).decode("utf-8")

    return f"data:{content_type};base64,{encoded}"


@app.get("/")
def home():
    try:
        return render_template("index.html")
    except Exception:
        return """
        <h1>VASUNDHARA VTON IS LIVE</h1>
        <p>Website is running.</p>
        """


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "message": "VASUNDHARA VTON SERVER IS RUNNING",
        "fashn_key_present": bool(FASHN_API_KEY)
    })


@app.get("/debug")
def debug():
    return jsonify({
        "status": "OK",
        "app_file": __file__,
        "fashn_key_present": bool(FASHN_API_KEY),
        "routes": [
            str(rule)
            for rule in app.url_map.iter_rules()
        ]
    })


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

        print("FASHN RUN STATUS:", response.status_code)
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

        # Poll until completed.
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

            print(
                "FASHN STATUS:",
                status_data.get("status")
            )

            status = status_data.get("status")

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
                    "prediction_id": prediction_id
                })

            if status in ["failed", "canceled", "cancelled", "error"]:

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

        print("❌ TRY-ON ERROR:", repr(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Flask 404",
        "requested_path": request.path
    }), 404


if __name__ == "__main__":

    port = int(os.environ.get("PORT", "8080"))

    app.run(
        host="0.0.0.0",
        port=port
    )
