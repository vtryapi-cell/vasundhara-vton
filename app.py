import base64
import os

import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

RUNPOD_ENDPOINT_URL = os.environ.get("RUNPOD_ENDPOINT_URL", "").strip()
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "").strip()
RUNPOD_TIMEOUT = int(os.environ.get("RUNPOD_TIMEOUT", "180"))


def image_to_base64(upload):
    if upload is None or not upload.filename:
        raise ValueError("Image upload is missing")
    raw = upload.read()
    if not raw:
        raise ValueError(f"Empty image upload: {upload.filename}")
    return base64.b64encode(raw).decode("utf-8")


def call_vasundhara(person_b64, saree_b64, seed=42, width=384, height=512):
    if not RUNPOD_ENDPOINT_URL:
        raise RuntimeError("RUNPOD_ENDPOINT_URL is not configured")
    if not RUNPOD_API_KEY:
        raise RuntimeError("RUNPOD_API_KEY is not configured")

    payload = {
        "input": {
            "person_image": person_b64,
            "saree_image": saree_b64,
            "seed": int(seed),
            "width": int(width),
            "height": int(height),
        }
    }

    response = requests.post(
        RUNPOD_ENDPOINT_URL,
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=RUNPOD_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("status") not in (None, "COMPLETED"):
        raise RuntimeError(f"RunPod job status: {data.get('status')}")

    output = data.get("output", data)
    if not isinstance(output, dict):
        raise RuntimeError("RunPod returned an invalid output")
    if not output.get("success", True):
        raise RuntimeError(output.get("error", "VASUNDHARA inference failed"))
    if not output.get("image"):
        raise RuntimeError("RunPod response did not contain an output image")

    return output


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "vasundhara-web-api",
        "runpod_configured": bool(RUNPOD_ENDPOINT_URL and RUNPOD_API_KEY),
    })


@app.post("/api/tryon")
def tryon():
    try:
        person = request.files.get("person") or request.files.get("person_image")
        saree = request.files.get("saree") or request.files.get("saree_image")
        if person is None:
            return jsonify({"success": False, "error": "Please upload a person photo."}), 400
        if saree is None:
            return jsonify({"success": False, "error": "Please upload a saree photo."}), 400

        seed = request.form.get("seed", 42)
        width = request.form.get("width", 384)
        height = request.form.get("height", 512)

        output = call_vasundhara(
            image_to_base64(person),
            image_to_base64(saree),
            seed=seed,
            width=width,
            height=height,
        )

        return jsonify({"success": True, **output})

    except requests.HTTPError as exc:
        body = exc.response.text[:1000] if exc.response is not None else str(exc)
        return jsonify({"success": False, "error": f"RunPod HTTP error: {body}"}), 502
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
