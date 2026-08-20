import os
import time
import base64
import requests

from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

# ============================================================
# API CONFIGURATION
# ============================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
FASHN_API_KEY = os.environ.get("FASHN_API_KEY")

OPENAI_URL = "https://api.openai.com/v1/images/edits"
FASHN_URL = "https://api.fashn.ai/v1"

print("========================================")
print("VASUNDHARA VTON APP STARTING")
print("========================================")
print("APP FILE:", __file__)
print("OPENAI KEY PRESENT:", bool(OPENAI_API_KEY))
print("FASHN KEY PRESENT:", bool(FASHN_API_KEY))
print("========================================")


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
        <p>Server is running.</p>
        """


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "message": "VASUNDHARA VTON SERVER IS RUNNING",
        "openai_key_present": bool(OPENAI_API_KEY),
        "fashn_key_present": bool(FASHN_API_KEY)
    })


# ============================================================
# DEBUG
# ============================================================

@app.get("/debug")
def debug():
    return jsonify({
        "status": "OK",
        "app_file": __file__,
        "openai_key_present": bool(OPENAI_API_KEY),
        "fashn_key_present": bool(FASHN_API_KEY),
        "routes": [
            str(rule)
            for rule in app.url_map.iter_rules()
        ]
    })


# ============================================================
# GET UPLOADED FILES
# ============================================================

def get_uploaded_files():

    model_file = request.files.get("model_image")

    if model_file is None:
        model_file = request.files.get("person")

    garment_file = request.files.get("garment_image")

    if garment_file is None:
        garment_file = request.files.get("garment")

    return model_file, garment_file


# ============================================================
# OPENAI SAREE TRY-ON
# ============================================================

@app.post("/api/openai-tryon")
def openai_tryon():

    print("")
    print("========================================")
    print("OPENAI TRY-ON REQUEST RECEIVED")
    print("========================================")

    # --------------------------------------------------------
    # CHECK OPENAI KEY
    # --------------------------------------------------------

    if not OPENAI_API_KEY:
        return jsonify({
            "success": False,
            "error": "OPENAI_API_KEY is not configured in Cloud Run."
        }), 500


    # --------------------------------------------------------
    # GET IMAGES
    # --------------------------------------------------------

    model_file, garment_file = get_uploaded_files()


    if model_file is None:
        return jsonify({
            "success": False,
            "error": "model_image is missing."
        }), 400


    if garment_file is None:
        return jsonify({
            "success": False,
            "error": "garment_image is missing."
        }), 400


    # --------------------------------------------------------
    # GET STYLE SELECTIONS
    # --------------------------------------------------------

    hairstyle = request.form.get(
        "hairstyle",
        "Natural"
    ).strip()

    background = request.form.get(
        "background",
        "Original"
    ).strip()


    print("SELECTED HAIRSTYLE:", hairstyle)
    print("SELECTED BACKGROUND:", background)


    # --------------------------------------------------------
    # READ MODEL IMAGE
    # --------------------------------------------------------

    try:

        model_data = model_file.read()

        if not model_data:
            return jsonify({
                "success": False,
                "error": "Model image is empty."
            }), 400


        # ----------------------------------------------------
        # READ SAREE IMAGE
        # ----------------------------------------------------

        garment_data = garment_file.read()

        if not garment_data:
            return jsonify({
                "success": False,
                "error": "Saree image is empty."
            }), 400


        print(
            "MODEL:",
            model_file.filename,
            len(model_data),
            "bytes"
        )

        print(
            "SAREE:",
            garment_file.filename,
            len(garment_data),
            "bytes"
        )


        # ====================================================
        # BUILD HAIRSTYLE INSTRUCTION
        # ====================================================

        if hairstyle.lower() in [
            "natural",
            "original"
        ]:

            hairstyle_instruction = """
Keep the person's original hairstyle exactly as shown
in IMAGE 1.

Do not change the hair.
"""

        else:

            hairstyle_instruction = f"""
Change the person's hairstyle to:

{hairstyle}

The requested hairstyle must be clearly visible
and professionally styled.

IMPORTANT:
- Keep the same person's face.
- Keep the same identity.
- Keep the same skin tone.
- Keep the same facial features.
- Do not change the person's age.
- Do not create a different person.
- Make the hairstyle realistic.
- Make the hairstyle naturally fit the person's head.
- Keep realistic hair texture and lighting.
"""


        # ====================================================
        # BUILD BACKGROUND INSTRUCTION
        # ====================================================

        if background.lower() in [
            "original",
            "keep photo"
        ]:

            background_instruction = """
Keep the original background from IMAGE 1.

Do not replace the background.
"""

        else:

            background_instruction = f"""
Replace the original background with:

{background}

The background should look like a professional
Indian fashion photography environment.

IMPORTANT:
- Keep the person clearly separated from the background.
- Keep realistic lighting.
- Add natural shadows.
- Make the background photorealistic.
- Do not distort the person.
- Do not change the saree.
"""


        # ====================================================
        # MAIN AI PROMPT
        # ====================================================

        prompt = f"""
Create a photorealistic Indian fashion virtual try-on
for a premium Indian saree e-commerce website.

IMAGE 1 is the PERSON / MODEL.

IMAGE 2 is the SAREE PRODUCT PHOTOGRAPH.

====================================================
SAREE
====================================================

Dress the exact person from IMAGE 1 in the exact saree
shown in IMAGE 2.

The final clothing MUST be a traditional Indian saree.

Do NOT turn the saree into:

- a dress
- a gown
- a skirt
- a jumpsuit
- a western one-piece
- a lehenga
- a salwar suit
- a kurti

Create realistic traditional Indian saree draping.

The saree must:

- wrap naturally around the waist
- cover the lower body
- have realistic front pleats
- have natural fabric folds
- have a realistic pallu
- place the pallu naturally over one shoulder
- look physically believable

====================================================
PRODUCT ACCURACY
====================================================

Use the EXACT saree product shown in IMAGE 2.

Preserve:

- exact saree colors
- exact border
- exact woven patterns
- exact decorative motifs
- exact pallu
- exact fabric appearance
- exact overall design

Do NOT invent another saree.

Do NOT replace the product with a generic saree.

Do NOT change the saree colors.

Do NOT change the saree pattern.

====================================================
PERSON ACCURACY
====================================================

Keep the SAME person from IMAGE 1.

Preserve:

- face
- facial features
- identity
- skin tone
- body proportions
- natural appearance

Do NOT create a different person.

Do NOT change the person's facial identity.

====================================================
HAIRSTYLE
====================================================

{hairstyle_instruction}

====================================================
BACKGROUND
====================================================

{background_instruction}

====================================================
REALISM
====================================================

Make the final result look like a professional
Indian fashion e-commerce photograph.

Use realistic:

- skin texture
- hair texture
- fabric texture
- lighting
- shadows
- proportions
- saree folds
- pallu placement

The saree must look naturally worn by the person.

The person must remain recognizable.

The final image should be photorealistic,
high quality and fashion-focused.

Do not add text.

Do not add logos.

Do not add watermarks.

====================================================
FINAL REQUIREMENT
====================================================

The final result must show:

1. The SAME person.
2. The EXACT saree from IMAGE 2.
3. The selected hairstyle: {hairstyle}.
4. The selected background: {background}.
5. Realistic Indian saree draping.
6. Professional fashion photography quality.
"""


        # ====================================================
        # FILE INFORMATION
        # ====================================================

        model_filename = (
            model_file.filename or "model.jpg"
        )

        garment_filename = (
            garment_file.filename or "saree.jpg"
        )

        model_content_type = (
            model_file.content_type or "image/jpeg"
        )

        garment_content_type = (
            garment_file.content_type or "image/jpeg"
        )


        # ====================================================
        # OPENAI MULTIPART REQUEST
        # ====================================================

        files = [

            (
                "image[]",
                (
                    model_filename,
                    model_data,
                    model_content_type
                )
            ),

            (
                "image[]",
                (
                    garment_filename,
                    garment_data,
                    garment_content_type
                )
            )

        ]


        data = {
            "model": "gpt-image-2",
            "prompt": prompt,
            "size": "1024x1536",
            "quality": "medium"
        }


        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }


        # ====================================================
        # CALL OPENAI
        # ====================================================

        print("🚀 Calling OpenAI image edit API...")
        print("HAIRSTYLE:", hairstyle)
        print("BACKGROUND:", background)

        start_time = time.time()

        response = requests.post(
            OPENAI_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=240
        )

        elapsed = time.time() - start_time


        print(
            "OPENAI STATUS:",
            response.status_code
        )

        print(
            "OPENAI TIME:",
            round(elapsed, 2),
            "seconds"
        )

        print(
            "OPENAI RESPONSE:",
            response.text[:5000]
        )


        # ====================================================
        # HANDLE OPENAI ERROR
        # ====================================================

        if not response.ok:

            try:
                error_json = response.json()
            except Exception:
                error_json = {}


            error_message = ""


            if isinstance(
                error_json.get("error"),
                dict
            ):

                error_message = (
                    error_json["error"].get("message")
                    or ""
                )


            if not error_message:

                error_message = (
                    response.text[:5000]
                    or "OpenAI image generation failed."
                )


            return jsonify({
                "success": False,
                "error": "OpenAI image generation failed.",
                "details": error_message,
                "openai_status": response.status_code
            }), 502


        # ====================================================
        # PARSE OPENAI RESPONSE
        # ====================================================

        try:

            result = response.json()

        except Exception as e:

            return jsonify({
                "success": False,
                "error": "OpenAI returned invalid JSON.",
                "details": str(e)
            }), 502


        # ====================================================
        # CHECK DATA
        # ====================================================

        if not result.get("data"):

            return jsonify({
                "success": False,
                "error": "OpenAI returned no image.",
                "response": result
            }), 502


        image_b64 = (
            result["data"][0].get("b64_json")
        )


        if not image_b64:

            return jsonify({
                "success": False,
                "error": "OpenAI returned no image data.",
                "response": result
            }), 502


        # ====================================================
        # RETURN IMAGE
        # ====================================================

        image_url = (
            "data:image/png;base64,"
            + image_b64
        )


        print("========================================")
        print("OPENAI RESULT READY")
        print("HAIRSTYLE:", hairstyle)
        print("BACKGROUND:", background)
        print("========================================")


        return jsonify({

            "success": True,

            "image_url": image_url,

            "provider": "openai",

            "model": "gpt-image-2",

            "hairstyle": hairstyle,

            "background": background

        })


    # ========================================================
    # ERRORS
    # ========================================================

    except requests.exceptions.Timeout:

        print("❌ OPENAI REQUEST TIMED OUT")

        return jsonify({
            "success": False,
            "error": "OpenAI request timed out."
        }), 504


    except requests.exceptions.RequestException as e:

        print(
            "❌ OPENAI NETWORK ERROR:",
            repr(e)
        )

        return jsonify({
            "success": False,
            "error": "Could not connect to OpenAI.",
            "details": str(e)
        }), 502


    except Exception as e:

        print(
            "❌ OPENAI UNEXPECTED ERROR:",
            repr(e)
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# MAIN TRY-ON ROUTE
# ============================================================

@app.post("/api/tryon")
def tryon():

    print("🔄 /api/tryon -> OpenAI")

    return openai_tryon()


# ============================================================
# OPTIONAL FASHN ROUTE
# ============================================================

@app.post("/api/fashn-tryon")
def fashn_tryon():

    if not FASHN_API_KEY:

        return jsonify({
            "success": False,
            "error": "FASHN_API_KEY is not configured."
        }), 500


    model_file, garment_file = get_uploaded_files()


    if model_file is None:

        return jsonify({
            "success": False,
            "error": "model_image is missing."
        }), 400


    if garment_file is None:

        return jsonify({
            "success": False,
            "error": "garment_image is missing."
        }), 400


    try:

        model_data = model_file.read()
        garment_data = garment_file.read()


        model_image = (
            "data:"
            + (
                model_file.content_type
                or "image/jpeg"
            )
            + ";base64,"
            + base64.b64encode(
                model_data
            ).decode("utf-8")
        )


        garment_image = (
            "data:"
            + (
                garment_file.content_type
                or "image/jpeg"
            )
            + ";base64,"
            + base64.b64encode(
                garment_data
            ).decode("utf-8")
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


        response = requests.post(

            f"{FASHN_URL}/run",

            json=payload,

            headers={

                "Authorization":
                    f"Bearer {FASHN_API_KEY}",

                "Content-Type":
                    "application/json"

            },

            timeout=60

        )


        if not response.ok:

            return jsonify({

                "success": False,

                "error": "FASHN request failed.",

                "details": response.text[:3000]

            }), 502


        run_data = response.json()

        prediction_id = run_data.get("id")


        if not prediction_id:

            return jsonify({

                "success": False,

                "error": "FASHN returned no prediction ID.",

                "response": run_data

            }), 502


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
                            "FASHN returned no output."

                    }), 502


                return jsonify({

                    "success": True,

                    "image_url": output[0],

                    "provider": "fashn",

                    "prediction_id":
                        prediction_id

                })


            if status in [
                "failed",
                "cancelled",
                "canceled",
                "error"
            ]:

                return jsonify({

                    "success": False,

                    "error":
                        "FASHN try-on failed.",

                    "details":
                        status_data.get("error")

                }), 502


        return jsonify({

            "success": False,

            "error":
                "FASHN prediction timed out."

        }), 504


    except Exception as e:

        print(
            "FASHN ERROR:",
            repr(e)
        )

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

        "success": False,

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
