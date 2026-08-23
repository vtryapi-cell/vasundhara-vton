import os
import time
import base64
import requests

from flask import Flask, jsonify, request, render_template


# ============================================================
# VASUNDHARA VTON - FASHN TRY-ON MAX
# ============================================================

app = Flask(__name__)

FASHN_API_KEY = os.environ.get("fa-vCE48mVgLYPn-A13fufqwI9FX39Iud5qNOpVO")

FASHN_URL = "https://api.fashn.ai/v1"


print("========================================")
print("VASUNDHARA VTON SERVER STARTING")
print("========================================")
print("APP FILE:", __file__)
print("FASHN KEY PRESENT:", bool(FASHN_API_KEY))
print("ENGINE: FASHN TRY-ON MAX")
print("========================================")


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    try:
        return render_template("index.html")

    except Exception as e:

        print("HOME ERROR:", repr(e))

        return """
        <html>
        <head>
            <title>Vasundhara VTON</title>
        </head>

        <body>

            <h1>VASUNDHARA VTON SERVER IS RUNNING</h1>

            <p>Template error:</p>

            <pre>{}</pre>

        </body>
        </html>
        """.format(str(e))


# ============================================================
# HEALTH
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({

        "ok": True,

        "message":
            "VASUNDHARA VTON SERVER IS RUNNING",

        "provider":
            "FASHN",

        "model":
            "tryon-max",

        "fashn_key_present":
            bool(FASHN_API_KEY)

    })


# ============================================================
# DEBUG
# ============================================================

@app.route("/debug", methods=["GET"])
def debug():

    return jsonify({

        "ok": True,

        "app_file":
            __file__,

        "fashn_key_present":
            bool(FASHN_API_KEY),

        "provider":
            "fashn",

        "model":
            "tryon-max",

        "routes": [
            str(rule)
            for rule in app.url_map.iter_rules()
        ]

    })


# ============================================================
# GET UPLOADED IMAGES
# ============================================================

def get_uploaded_files():

    model_file = request.files.get(
        "model_image"
    )

    if model_file is None:

        model_file = request.files.get(
            "person"
        )

    garment_file = request.files.get(
        "garment_image"
    )

    if garment_file is None:

        garment_file = request.files.get(
            "garment"
        )

    return model_file, garment_file


# ============================================================
# CONVERT FILE TO DATA URI
# ============================================================

def file_to_data_uri(file_object):

    data = file_object.read()

    if not data:
        raise ValueError("Uploaded image is empty.")

    content_type = (
        file_object.content_type
        or "image/jpeg"
    )

    encoded = base64.b64encode(
        data
    ).decode("utf-8")

    return (
        f"data:{content_type};base64,{encoded}"
    )


# ============================================================
# FASHN TRY-ON MAX
# ============================================================

@app.route(
    "/api/fashn-tryon",
    methods=["POST"]
)
def fashn_tryon():

    print("")
    print("========================================")
    print("FASHN TRY-ON MAX REQUEST")
    print("========================================")


    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    if not FASHN_API_KEY:

        return jsonify({

            "success": False,

            "error":
                "FASHN_API_KEY is not configured."

        }), 500


    # --------------------------------------------------------
    # FILES
    # --------------------------------------------------------

    model_file, garment_file = (
        get_uploaded_files()
    )


    if model_file is None:

        return jsonify({

            "success": False,

            "error":
                "model_image is missing."

        }), 400


    if garment_file is None:

        return jsonify({

            "success": False,

            "error":
                "garment_image is missing."

        }), 400


    try:

        # ----------------------------------------------------
        # READ IMAGES
        # ----------------------------------------------------

        model_image = file_to_data_uri(
            model_file
        )

        garment_image = file_to_data_uri(
            garment_file
        )


        print(
            "MODEL:",
            model_file.filename
        )

        print(
            "SAREE:",
            garment_file.filename
        )


        # ----------------------------------------------------
        # OPTIONS FROM WEBSITE
        # ----------------------------------------------------

        hairstyle = request.form.get(
            "hairstyle",
            "Original"
        ).strip()


        background = request.form.get(
            "background",
            "Original"
        ).strip()


        # ----------------------------------------------------
        # QUALITY
        # ----------------------------------------------------

        resolution = request.form.get(
            "resolution",
            "2k"
        ).strip().lower()


        if resolution not in [
            "1k",
            "2k",
            "4k"
        ]:

            resolution = "2k"


        generation_mode = request.form.get(
            "generation_mode",
            "quality"
        ).strip().lower()


        if generation_mode not in [
            "fast",
            "balanced",
            "quality"
        ]:

            generation_mode = "quality"


        # ----------------------------------------------------
        # NUMBER OF IMAGES
        # ----------------------------------------------------

        try:

            num_images = int(
                request.form.get(
                    "num_images",
                    "1"
                )
            )

        except Exception:

            num_images = 1


        num_images = max(
            1,
            min(num_images, 4)
        )


        # ----------------------------------------------------
        # SEED
        # ----------------------------------------------------

        try:

            seed = int(
                request.form.get(
                    "seed",
                    "42"
                )
            )

        except Exception:

            seed = 42


        # ----------------------------------------------------
        # SAREE PROMPT
        # ----------------------------------------------------

        prompt = """
Create a highly realistic Indian bridal saree
virtual try-on.

Use the EXACT person from the model image.

Preserve the person's:

- face
- facial identity
- facial structure
- skin tone
- body proportions
- pose
- hands
- natural appearance

Do not create a different person.

Use the EXACT saree/product shown in the
product image.

Preserve the saree's:

- original color
- original fabric
- original pattern
- original motifs
- original border
- original pallu
- original decorative details

Do not replace the saree with a generic saree.

The product must remain visually faithful
to the supplied saree.

Dress the person in a traditional
Indian saree.

IMPORTANT SAREE DRAPING:

The saree should look physically worn,
professionally draped and naturally fitted.

Create a traditional Indian front drape
with clearly visible, compact,
well-organized vertical saree pleats.

The front pleats should be:

- narrow
- numerous
- compact
- vertical
- evenly arranged
- neatly folded
- professionally pressed
- gathered naturally at the waist
- falling naturally downward

Do not make the lower body look like
a skirt or gown.

Do not spread the saree fabric into
one large flat panel.

Keep the saree border visible.

Keep the pallu separate from the
front pleats and drape it naturally
over the shoulder.

The final result should look like a
real Indian bridal fashion photograph.

Maintain realistic:

- fabric folds
- fabric tension
- shadows
- lighting
- skin texture
- hair
- anatomy
- hands
- proportions

Do not add text.

Do not add logos.

Do not add watermarks.

Do not change the person's identity.

Do not invent a different saree.
"""


        # ----------------------------------------------------
        # HAIRSTYLE
        # ----------------------------------------------------

        if hairstyle.lower() not in [
            "original",
            "natural",
            "keep original"
        ]:

            prompt += f"""

HAIRSTYLE:

Apply this requested hairstyle:

{hairstyle}

Keep the SAME person's face and identity.

The hairstyle must look realistic,
natural and suitable for an Indian
bridal fashion photograph.

Do not alter facial identity.
"""


        else:

            prompt += """

HAIRSTYLE:

Keep the person's original hairstyle.

Do not unnecessarily change the hair.
"""


        # ----------------------------------------------------
        # BACKGROUND
        # ----------------------------------------------------

        if background.lower() not in [
            "original",
            "keep photo",
            "keep original"
        ]:

            prompt += f"""

BACKGROUND:

Use this background:

{background}

Keep the person and saree realistic.

Use natural depth, lighting and shadows.

Do not distort the person or saree.
"""

        else:

            prompt += """

BACKGROUND:

Keep the original background
from the model image.
"""


        # ----------------------------------------------------
        # FINAL PROMPT
        # ----------------------------------------------------

        print("")
        print("HAIRSTYLE:", hairstyle)
        print("BACKGROUND:", background)
        print("RESOLUTION:", resolution)
        print("MODE:", generation_mode)
        print("NUM IMAGES:", num_images)
        print("SEED:", seed)
        print("========================================")


        # ----------------------------------------------------
        # FASHN TRY-ON MAX PAYLOAD
        # ----------------------------------------------------

        payload = {

            "model_name":
                "tryon-max",

            "inputs": {

                "model_image":
                    model_image,

                "product_image":
                    garment_image,

                "prompt":
                    prompt,

                "resolution":
                    resolution,

                "generation_mode":
                    generation_mode,

                "num_images":
                    num_images,

                "output_format":
                    "png",

                "return_base64":
                    False,

                "seed":
                    seed

            }

        }


        # ----------------------------------------------------
        # SEND REQUEST
        # ----------------------------------------------------

        print("")
        print("🚀 CALLING FASHN TRY-ON MAX...")


        start_time = time.time()


        response = requests.post(

            f"{FASHN_URL}/run",

            json=payload,

            headers={

                "Authorization":
                    f"Bearer {FASHN_API_KEY}",

                "Content-Type":
                    "application/json"

            },

            timeout=120

        )


        elapsed = (
            time.time()
            - start_time
        )


        print(
            "FASHN STATUS:",
            response.status_code
        )

        print(
            "REQUEST TIME:",
            round(elapsed, 2),
            "seconds"
        )


        # ----------------------------------------------------
        # API ERROR
        # ----------------------------------------------------

        if not response.ok:

            try:

                error_data = (
                    response.json()
                )

            except Exception:

                error_data = {
                    "raw":
                        response.text[:5000]
                }


            print(
                "FASHN ERROR:",
                error_data
            )


            return jsonify({

                "success":
                    False,

                "error":
                    "FASHN request failed.",

                "details":
                    error_data,

                "status":
                    response.status_code

            }), 502


        # ----------------------------------------------------
        # PARSE
        # ----------------------------------------------------

        try:

            run_data = (
                response.json()
            )

        except Exception as e:

            return jsonify({

                "success":
                    False,

                "error":
                    "FASHN returned invalid JSON.",

                "details":
                    str(e)

            }), 502


        prediction_id = (
            run_data.get("id")
        )


        if not prediction_id:

            return jsonify({

                "success":
                    False,

                "error":
                    "FASHN returned no prediction ID.",

                "response":
                    run_data

            }), 502


        print(
            "PREDICTION ID:",
            prediction_id
        )


        # ----------------------------------------------------
        # POLL
        # ----------------------------------------------------

        for attempt in range(60):

            time.sleep(3)


            status_response = requests.get(

                f"{FASHN_URL}/status/{prediction_id}",

                headers={

                    "Authorization":
                        f"Bearer {FASHN_API_KEY}"

                },

                timeout=60

            )


            try:

                status_data = (
                    status_response.json()
                )

            except Exception:

                status_data = {}


            status = (
                status_data.get(
                    "status"
                )
            )


            print(
                "FASHN STATUS:",
                status,
                "| ATTEMPT:",
                attempt + 1
            )


            # ------------------------------------------------
            # COMPLETED
            # ------------------------------------------------

            if status == "completed":

                output = (
                    status_data.get(
                        "output",
                        []
                    )
                )


                if not output:

                    return jsonify({

                        "success":
                            False,

                        "error":
                            "FASHN completed but returned no image."

                    }), 502


                print("")
                print("========================================")
                print("✅ FASHN TRY-ON COMPLETE")
                print("IMAGES:", len(output))
                print("========================================")


                return jsonify({

                    "success":
                        True,

                    "provider":
                        "fashn",

                    "model":
                        "tryon-max",

                    "prediction_id":
                        prediction_id,

                    "image_url":
                        output[0],

                    "images":
                        output,

                    "hairstyle":
                        hairstyle,

                    "background":
                        background,

                    "resolution":
                        resolution,

                    "generation_mode":
                        generation_mode,

                    "num_images":
                        len(output)

                })


            # ------------------------------------------------
            # FAILED
            # ------------------------------------------------

            if status in [
                "failed",
                "cancelled",
                "canceled",
                "error"
            ]:

                print(
                    "FASHN FAILED:",
                    status_data
                )


                return jsonify({

                    "success":
                        False,

                    "error":
                        "FASHN try-on failed.",

                    "details":
                        status_data.get(
                            "error",
                            status_data
                        ),

                    "prediction_id":
                        prediction_id

                }), 502


        # ----------------------------------------------------
        # TIMEOUT
        # ----------------------------------------------------

        return jsonify({

            "success":
                False,

            "error":
                "FASHN prediction timed out.",

            "prediction_id":
                prediction_id

        }), 504


    # ========================================================
    # REQUEST ERROR
    # ========================================================

    except requests.exceptions.Timeout:

        print(
            "❌ FASHN REQUEST TIMEOUT"
        )

        return jsonify({

            "success":
                False,

            "error":
                "FASHN request timed out."

        }), 504


    except requests.exceptions.RequestException as e:

        print(
            "❌ FASHN NETWORK ERROR:",
            repr(e)
        )

        return jsonify({

            "success":
                False,

            "error":
                "Could not connect to FASHN.",

            "details":
                str(e)

        }), 502


    except Exception as e:

        print(
            "❌ UNEXPECTED ERROR:",
            repr(e)
        )

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ============================================================
# MAIN TRY-ON ROUTE
# ============================================================

@app.route(
    "/api/tryon",
    methods=["POST"]
)
def tryon():

    print(
        "🔄 /api/tryon -> FASHN TRY-ON MAX"
    )

    return fashn_tryon()


# ============================================================
# 404
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "success":
            False,

        "error":
            "Flask 404",

        "requested_path":
            request.path

    }), 404


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "8080"
        )
    )


    print(
        "Starting server on port",
        port
    )


    app.run(

        host="0.0.0.0",

        port=port

    )
