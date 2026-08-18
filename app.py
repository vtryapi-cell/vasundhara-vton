import os
import time
import base64
import mimetypes
import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB total request

FASHN_API_KEY = os.environ.get("FASHN_API_KEY")
FASHN_BASE_URL = "https://api.fashn.ai/v1"

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

def image_to_data_uri(file_storage):
    content_type = file_storage.mimetype or "image/jpeg"
    if content_type not in ALLOWED_TYPES:
        raise ValueError("Only JPG, PNG, and WebP images are supported.")
    raw = file_storage.read()
    if not raw:
        raise ValueError("One of the uploaded images is empty.")
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({"ok": True, "api_key_configured": bool(FASHN_API_KEY)})

@app.post("/api/tryon")
def tryon():
    if not FASHN_API_KEY:
        return jsonify({"error": "FASHN_API_KEY is not configured on the server."}), 500

    person = request.files.get("person")
    garment = request.files.get("garment")

    if not person or not garment:
        return jsonify({"error": "Please upload both your photo and garment photo."}), 400

    try:
        model_image = image_to_data_uri(person)
        garment_image = image_to_data_uri(garment)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    payload = {
        "model_name": "tryon-v1.6",
        "inputs": {
            "model_image": model_image,
            "garment_image": garment_image,
            "category": "auto",
            "garment_photo_type": "auto",
            "mode": "balanced",
            "num_samples": 1,
            "output_format": "png"
        }
    }

    headers = {
        "Authorization": f"Bearer {FASHN_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        run = requests.post(
            f"{FASHN_BASE_URL}/run",
            headers=headers,
            json=payload,
            timeout=60,
        )
        run.raise_for_status()
        run_data = run.json()
        prediction_id = run_data.get("id")
        if not prediction_id:
            return jsonify({"error": run_data.get("error") or "FASHN did not return a prediction ID."}), 502

        # Poll for up to ~2 minutes. This is one prediction request plus status checks.
        for _ in range(40):
            time.sleep(3)
            status = requests.get(
                f"{FASHN_BASE_URL}/status/{prediction_id}",
                headers={"Authorization": f"Bearer {FASHN_API_KEY}"},
                timeout=30,
            )
            status.raise_for_status()
            data = status.json()
            state = data.get("status")

            if state == "completed":
                output = data.get("output")
                if isinstance(output, list) and output:
                    return jsonify({"success": True, "image_url": output[0]})
                return jsonify({"error": "Prediction completed but no output image was returned."}), 502

            if state not in {"starting", "in_queue", "processing"}:
                return jsonify({"error": data.get("error") or f"Prediction ended with status: {state}"}), 502

        return jsonify({"error": "The try-on is taking longer than expected. Please try again."}), 504

    except requests.HTTPError:
        detail = ""
        try:
            detail = run.text[:500]
        except Exception:
            pass
        return jsonify({"error": f"FASHN API error. {detail}"}), 502
    except requests.RequestException as exc:
        return jsonify({"error": f"Could not reach the FASHN API: {exc}"}), 502

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
