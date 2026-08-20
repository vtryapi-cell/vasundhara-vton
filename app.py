import os
import time
import base64
import requests

from flask import Flask, jsonify, request, render_template


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# API CONFIGURATION
# ============================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
FASHN_API_KEY = os.environ.get("FASHN_API_KEY")

OPENAI_URL = "https://api.openai.com/v1/images/edits"
FASHN_URL = "https://api.fashn.ai/v1"


print("========================================")
print("VASUNDHARA VTON SERVER STARTING")
print("========================================")
print("APP FILE:", __file__)
print("OPENAI KEY PRESENT:", bool(OPENAI_API_KEY))
print("FASHN KEY PRESENT:", bool(FASHN_API_KEY))
print("========================================")


# ============================================================
# HOME PAGE
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
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({

        "ok": True,

        "message":
            "VASUNDHARA VTON SERVER IS RUNNING",

        "openai_key_present":
            bool(OPENAI_API_KEY),

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

        "openai_key_present":
            bool(OPENAI_API_KEY),

        "fashn_key_present":
            bool(FASHN_API_KEY),

        "routes": [
            str(rule)
            for rule in app.url_map.iter_rules()
        ]

    })


# ============================================================
# GET UPLOADED FILES
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
# OPENAI TRY-ON
# ============================================================

@app.route(
    "/api/openai-tryon",
    methods=["POST"]
)
def openai_tryon():

    print("")
    print("========================================")
    print("OPENAI TRY-ON REQUEST")
    print("========================================")


    # --------------------------------------------------------
    # CHECK API KEY
    # --------------------------------------------------------

    if not OPENAI_API_KEY:

        return jsonify({

            "success": False,

            "error":
                "OPENAI_API_KEY is not configured."

        }), 500


    # --------------------------------------------------------
    # GET FILES
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


    # --------------------------------------------------------
    # GET HAIRSTYLE
    # --------------------------------------------------------

    hairstyle = request.form.get(
        "hairstyle",
        "Natural"
    ).strip()


    # --------------------------------------------------------
    # GET BACKGROUND
    # --------------------------------------------------------

    background = request.form.get(
        "background",
        "Original"
    ).strip()


    print(
        "SELECTED HAIRSTYLE:",
        hairstyle
    )

    print(
        "SELECTED BACKGROUND:",
        background
    )


    # --------------------------------------------------------
    # READ IMAGES
    # --------------------------------------------------------

    try:

        model_data = model_file.read()

        garment_data = garment_file.read()


        if not model_data:

            return jsonify({

                "success": False,

                "error":
                    "Model image is empty."

            }), 400


        if not garment_data:

            return jsonify({

                "success": False,

                "error":
                    "Saree image is empty."

            }), 400


        print(
            "MODEL IMAGE:",
            model_file.filename,
            len(model_data),
            "bytes"
        )


        print(
            "SAREE IMAGE:",
            garment_file.filename,
            len(garment_data),
            "bytes"
        )


        # ====================================================
        # HAIRSTYLE INSTRUCTION
        # ====================================================

        if hairstyle.lower() in [
            "natural",
            "original"
        ]:

            hairstyle_instruction = """

HAIRSTYLE:
Keep the person's original hairstyle
exactly as shown in IMAGE 1.

Do not change the hair.
Do not alter the face.
Do not alter the identity.

"""

        else:

            hairstyle_instruction = f"""

HAIRSTYLE:
The selected hairstyle is:

{hairstyle}

Change the person's hairstyle to this
selected hairstyle.

The hairstyle must be clearly visible.

IMPORTANT:

- Keep the SAME person.
- Keep the SAME face.
- Keep the SAME facial features.
- Keep the SAME skin tone.
- Keep the SAME age.
- Do not create a different person.
- Do not change facial identity.
- Make the hairstyle realistic.
- Make the hairstyle naturally fit the head.
- Use realistic Indian hair texture.
- Use realistic lighting on the hair.
- Make it look like professional bridal/fashion styling.

"""


        # ====================================================
        # BACKGROUND INSTRUCTION
        # ====================================================

        if background.lower() in [
            "original",
            "keep photo"
        ]:

            background_instruction = """

BACKGROUND:
Keep the original background
from IMAGE 1.

Do not replace the background.

"""

        else:

            background_instruction = f"""

BACKGROUND:
Replace the original background with:

{background}

The background must look like a
professional Indian fashion photograph.

IMPORTANT:

- Keep the person clearly visible.
- Keep the person separated from the background.
- Use realistic depth.
- Use realistic shadows.
- Use realistic lighting.
- Do not distort the person.
- Do not distort the saree.
- Do not add text.
- Do not add logos.
- Make the environment photorealistic.

"""


        # ====================================================
        # MAIN PROMPT
        # ====================================================

        prompt = f"""

You are creating a PREMIUM INDIAN BRIDAL SAREE
VIRTUAL TRY-ON IMAGE.

IMAGE 1 = PERSON / MODEL

IMAGE 2 = SAREE PRODUCT IMAGE


============================================================
PERSON
============================================================

Use the EXACT PERSON from IMAGE 1.

Preserve:

- face
- facial identity
- facial features
- skin tone
- body proportions
- natural appearance
- realistic hands
- realistic arms
- realistic body shape

DO NOT create a different person.

DO NOT change the person's identity.

DO NOT change the person's facial structure.


============================================================
SAREE PRODUCT
============================================================

Use the EXACT SAREE from IMAGE 2.

Preserve the actual:

- saree color
- saree pattern
- saree motifs
- saree border
- saree pallu
- fabric appearance
- woven design
- decorative details

DO NOT invent another saree.

DO NOT change the saree color.

DO NOT replace the saree with a generic saree.

DO NOT turn it into a:

- gown
- dress
- skirt
- lehenga
- salwar suit
- kurti
- western outfit
- jumpsuit


============================================================
TRADITIONAL INDIAN SAREE DRAPING
============================================================

Dress the person in the exact saree.

The saree must be worn as a
traditional Indian saree.

The saree must:

- wrap naturally around the waist
- cover the lower body
- have realistic fabric tension
- have realistic folds
- have realistic border placement
- have a natural pallu
- have a proper shoulder drape
- look physically worn
- look like professionally draped Indian bridal clothing


============================================================
BRIDAL FRONT PLEATS
============================================================

THIS IS EXTREMELY IMPORTANT.

The FRONT SAREE PLEATS MUST BE
PROPERLY FOLDED IN A TRADITIONAL
INDIAN BRIDAL STYLE.

The front pleats must be:

- narrow
- numerous
- compact
- vertical
- evenly arranged
- tightly folded
- neatly stacked
- symmetrical
- professionally pressed
- clearly visible
- starting from the waist
- falling vertically downward

The pleats must remain together
in the center/front of the lower body.

DO NOT leave loose fabric in the front.

DO NOT leave a large unfolded sheet
of saree fabric in front.

DO NOT make the saree look like a skirt.

DO NOT spread the pleats widely apart.

DO NOT leave loose fabric hanging
between the legs.

DO NOT make random fabric folds.

DO NOT create one large flat fabric panel.

DO NOT leave the saree unfolded.

The front pleats should look like
a professional Indian bridal saree
has been carefully hand-folded and tucked.

The bottom saree border should remain
visible across the folded pleats.

The pleats must preserve the exact
border and motifs from IMAGE 2.

The pallu must remain separate from
the front pleats.

The pallu should fall naturally over
the shoulder.


============================================================
BRIDAL FINISH
============================================================

Make the saree look like it has been
professionally draped for an Indian wedding.

The overall appearance should be:

- elegant
- traditional
- bridal
- premium
- realistic
- photographic

Keep the saree neat and structured.

The front pleats are a priority.


============================================================
HAIRSTYLE
============================================================

{hairstyle_instruction}


============================================================
BACKGROUND
============================================================

{background_instruction}


============================================================
REALISM
============================================================

The final image must look like a
professional Indian fashion e-commerce photograph.

Use realistic:

- skin texture
- hair texture
- fabric texture
- saree folds
- pleat structure
- lighting
- shadows
- proportions
- anatomy
- hands
- jewelry
- fabric tension

Do not produce cartoon-like results.

Do not produce painted results.

Do not add text.

Do not add logos.

Do not add watermarks.


============================================================
FINAL REQUIREMENTS
============================================================

The final image MUST contain:

1. The SAME person.
2. The EXACT saree from IMAGE 2.
3. Traditional Indian saree draping.
4. CLEARLY FOLDED bridal front pleats.
5. Compact vertical pleats.
6. No loose unfolded front fabric.
7. Correct saree border placement.
8. Natural pallu.
9. Selected hairstyle: {hairstyle}.
10. Selected background: {background}.
11. Photorealistic fashion photography.
12. Premium Indian bridal appearance.

PRIORITY ORDER:

1. Preserve person identity.
2. Preserve exact saree design.
3. Create properly folded bridal front pleats.
4. Apply selected hairstyle.
5. Apply selected background.
6. Maintain photorealistic quality.

"""


        print("")
        print("========================================")
        print("AI PROMPT PREPARED")
        print("HAIRSTYLE:", hairstyle)
        print("BACKGROUND:", background)
        print("BRIDAL PLEATS: ENABLED")
        print("========================================")


        # ====================================================
        # FILE NAMES
        # ====================================================

        model_filename = (
            model_file.filename
            or "model.jpg"
        )

        garment_filename = (
            garment_file.filename
            or "saree.jpg"
        )


        model_content_type = (
            model_file.content_type
            or "image/jpeg"
        )

        garment_content_type = (
            garment_file.content_type
            or "image/jpeg"
        )


        # ====================================================
        # OPENAI FILES
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


        # ====================================================
        # OPENAI REQUEST DATA
        # ====================================================

        data = {

            "model":
                "gpt-image-1-mini",

            "prompt":
                prompt,

            "size":
                "1024x1536",

            "quality":
                "low"

        }


        headers = {

            "Authorization":
                f"Bearer {OPENAI_API_KEY}"

        }


        # ====================================================
        # CALL OPENAI
        # ====================================================

        print("")
        print("🚀 CALLING OPENAI...")
        print(
            "HAIRSTYLE:",
            hairstyle
        )
        print(
            "BACKGROUND:",
            background
        )
        print(
            "BRIDAL PLEATS: YES"
        )


        start_time = time.time()


        response = requests.post(

            OPENAI_URL,

            headers=headers,

            files=files,

            data=data,

            timeout=240

        )


        elapsed = (
            time.time()
            - start_time
        )


        print(
            "OPENAI STATUS:",
            response.status_code
        )

        print(
            "OPENAI TIME:",
            round(elapsed, 2),
            "seconds"
        )


        # ====================================================
        # OPENAI ERROR
        # ====================================================

        if not response.ok:

            try:

                error_json = (
                    response.json()
                )

            except Exception:

                error_json = {}


            error_message = ""


            if isinstance(
                error_json.get("error"),
                dict
            ):

                error_message = (
                    error_json["error"]
                    .get("message", "")
                )


            if not error_message:

                error_message = (
                    response.text[:5000]
                )


            print(
                "OPENAI ERROR:",
                error_message
            )


            return jsonify({

                "success":
                    False,

                "error":
                    "OpenAI image generation failed.",

                "details":
                    error_message,

                "openai_status":
                    response.status_code

            }), 502


        # ====================================================
        # PARSE RESPONSE
        # ====================================================

        try:

            result = (
                response.json()
            )

        except Exception as e:

            print(
                "JSON ERROR:",
                repr(e)
            )

            return jsonify({

                "success":
                    False,

                "error":
                    "OpenAI returned invalid JSON.",

                "details":
                    str(e)

            }), 502


        # ====================================================
        # CHECK IMAGE DATA
        # ====================================================

        if not result.get("data"):

            print(
                "NO IMAGE DATA:",
                result
            )

            return jsonify({

                "success":
                    False,

                "error":
                    "OpenAI returned no image.",

                "response":
                    result

            }), 502


        image_b64 = (
            result["data"][0]
            .get("b64_json")
        )


        if not image_b64:

            print(
                "NO B64 IMAGE DATA"
            )

            return jsonify({

                "success":
                    False,

                "error":
                    "OpenAI returned no image data.",

                "response":
                    result

            }), 502


        # ====================================================
        # IMAGE URL
        # ====================================================

        image_url = (
            "data:image/png;base64,"
            + image_b64
        )


        print("")
        print("========================================")
        print("✅ IMAGE GENERATED")
        print("HAIRSTYLE:", hairstyle)
        print("BACKGROUND:", background)
        print("BRIDAL PLEATS: YES")
        print("========================================")


        # ====================================================
        # RETURN RESULT
        # ====================================================

        return jsonify({

            "success":
                True,

            "image_url":
                image_url,

            "provider":
                "openai",

            "model":
                "gpt-image-2",

            "hairstyle":
                hairstyle,

            "background":
                background,

            "bridal_pleats":
                True

        })


    # ========================================================
    # TIMEOUT
    # ========================================================

    except requests.exceptions.Timeout:

        print(
            "❌ OPENAI TIMEOUT"
        )

        return jsonify({

            "success":
                False,

            "error":
                "OpenAI request timed out."

        }), 504


    # ========================================================
    # NETWORK ERROR
    # ========================================================

    except requests.exceptions.RequestException as e:

        print(
            "❌ OPENAI NETWORK ERROR:",
            repr(e)
        )

        return jsonify({

            "success":
                False,

            "error":
                "Could not connect to OpenAI.",

            "details":
                str(e)

        }), 502


    # ========================================================
    # GENERAL ERROR
    # ========================================================

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
        "🔄 /api/tryon -> OpenAI"
    )

    return openai_tryon()


# ============================================================
# OPTIONAL FASHN TRY-ON
# ============================================================

@app.route(
    "/api/fashn-tryon",
    methods=["POST"]
)
def fashn_tryon():

    if not FASHN_API_KEY:

        return jsonify({

            "success":
                False,

            "error":
                "FASHN_API_KEY is not configured."

        }), 500


    model_file, garment_file = (
        get_uploaded_files()
    )


    if model_file is None:

        return jsonify({

            "success":
                False,

            "error":
                "model_image is missing."

        }), 400


    if garment_file is None:

        return jsonify({

            "success":
                False,

            "error":
                "garment_image is missing."

        }), 400


    try:

        model_data = (
            model_file.read()
        )

        garment_data = (
            garment_file.read()
        )


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

            "model_name":
                "tryon-v1.6",

            "inputs": {

                "model_image":
                    model_image,

                "garment_image":
                    garment_image,

                "category":
                    "auto",

                "garment_photo_type":
                    "auto",

                "mode":
                    "balanced",

                "num_samples":
                    1,

                "output_format":
                    "jpeg"

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

                "success":
                    False,

                "error":
                    "FASHN request failed.",

                "details":
                    response.text[:3000]

            }), 502


        run_data = (
            response.json()
        )


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


            status_data = (
                status_response.json()
            )


            status = (
                status_data.get("status")
            )


            print(
                "FASHN STATUS:",
                status
            )


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
                            "FASHN returned no output."

                    }), 502


                return jsonify({

                    "success":
                        True,

                    "image_url":
                        output[0],

                    "provider":
                        "fashn",

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

                    "success":
                        False,

                    "error":
                        "FASHN try-on failed.",

                    "details":
                        status_data.get(
                            "error"
                        )

                }), 502


        return jsonify({

            "success":
                False,

            "error":
                "FASHN prediction timed out."

        }), 504


    except Exception as e:

        print(
            "FASHN ERROR:",
            repr(e)
        )

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ============================================================
# 404 HANDLER
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
