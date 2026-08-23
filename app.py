import os
import time
import base64
import requests

from flask import Flask, jsonify, request, render_template


# ============================================================
# VASUNDHARA VTON
# FASHN TRY-ON MAX
# ============================================================

app = Flask(__name__)

FASHN_API_KEY = os.environ.get("FASHN_API_KEY")

FASHN_URL = "https://api.fashn.ai/v1"


print("========================================")
print("VASUNDHARA VTON SERVER")
print("========================================")
print("APP FILE:", __file__)
print("FASHN KEY PRESENT:", bool(FASHN_API_KEY))
print("ENGINE: FASHN TRY-ON MAX")
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
        <!DOCTYPE html>
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

        "provider":
            "FASHN",

        "model":
            "tryon-max",

        "fashn_key_present":
            bool(FASHN_API_KEY),

        "routes": [

            str(rule)

            for rule in app.url_map.iter_rules()

        ]

    })


# ============================================================
# GET MODEL + SAREE FILES
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
# FILE -> DATA URI
# ============================================================

def file_to_data_uri(file_object):

    data = file_object.read()

    if not data:

        raise ValueError(
            "Uploaded image is empty."
        )

    content_type = (
        file_object.content_type
        or "image/jpeg"
    )

    encoded = base64.b64encode(
        data
    ).decode("utf-8")

    return (
        "data:"
        + content_type
        + ";base64,"
        + encoded
    )


# ============================================================
# MAIN VTON
# ============================================================

@app.route(
    "/api/tryon",
    methods=["POST"]
)
def tryon():

    print("")
    print("========================================")
    print("NEW VTON REQUEST")
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


    try:

        # ----------------------------------------------------
        # FILE INFORMATION
        # ----------------------------------------------------

        print(
            "MODEL FILE:",
            model_file.filename
        )

        print(
            "SAREE FILE:",
            garment_file.filename
        )


        # ----------------------------------------------------
        # CONVERT IMAGES
        # ----------------------------------------------------

        model_image = file_to_data_uri(
            model_file
        )

        garment_image = file_to_data_uri(
            garment_file
        )


        # ----------------------------------------------------
        # OPTIONAL WEBSITE SETTINGS
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
        # RESOLUTION
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


        # ----------------------------------------------------
        # GENERATION MODE
        # ----------------------------------------------------

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


        if num_images < 1:

            num_images = 1


        if num_images > 4:

            num_images = 4


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


        if seed < 0:

            seed = 42


        # ----------------------------------------------------
        # PROMPT
        #
        # IMPORTANT:
        # KEEP THIS VERY SHORT.
        #
        # We do NOT ask the AI to create a new model,
        # new background, new face, new hairstyle, etc.
        # ----------------------------------------------------

        prompt = request.form.get(
            "prompt",
            ""
        ).strip()


        # ----------------------------------------------------
        # DEFAULT SAREE INSTRUCTION
        # ----------------------------------------------------

        if not prompt:

            prompt = (
                "Apply the supplied saree naturally "
                "to the person. Preserve the person's "
                "identity, pose, body proportions and "
                "original appearance. Preserve the "
                "saree's original color, pattern, border, "
                "motifs and pallu. Use a traditional "
                "Indian saree drape with neat compact "
                "front pleats and a natural shoulder pallu."
            )


        # ----------------------------------------------------
        # PRINT SETTINGS
        # ----------------------------------------------------

        print("")
        print("----------------------------------------")
        print("PROVIDER: FASHN")
        print("MODEL: tryon-max")
        print("RESOLUTION:", resolution)
        print("MODE:", generation_mode)
        print("NUM IMAGES:", num_images)
        print("SEED:", seed)
        print("HAIRSTYLE:", hairstyle)
        print("BACKGROUND:", background)
        print("PROMPT:", prompt)
        print("----------------------------------------")


        # ====================================================
        # FASHN REQUEST
        # ====================================================

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

                "seed":
                    seed,

                "num_images":
                    num_images,

                "output_format":
                    "png",

                "return_base64":
                    False

            }

        }


        # ----------------------------------------------------
        # START REQUEST
        # ----------------------------------------------------

        print("")
        print("🚀 CALLING FASHN TRY-ON MAX...")


        start_time = time.time()


        response = requests.post(

            FASHN_URL + "/run",

            json=payload,

            headers={

                "Authorization":
                    "Bearer " + FASHN_API_KEY,

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
            "FASHN HTTP STATUS:",
            response.status_code
        )

        print(
            "REQUEST TIME:",
            round(elapsed, 2),
            "seconds"
        )


        # ----------------------------------------------------
        # FASHN API ERROR
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

                "success": False,

                "error":
                    "FASHN request failed.",

                "details":
                    error_data,

                "status":
                    response.status_code

            }), 502


        # ----------------------------------------------------
        # PARSE START RESPONSE
        # ----------------------------------------------------

        try:

            run_data = (
                response.json()
            )

        except Exception as e:

            return jsonify({

                "success": False,

                "error":
                    "FASHN returned invalid JSON.",

                "details":
                    str(e)

            }), 502


        prediction_id = (
            run_data.get("id")
        )


        if not prediction_id:

            print(
                "NO PREDICTION ID:",
                run_data
            )


            return jsonify({

                "success": False,

                "error":
                    "FASHN returned no prediction ID.",

                "response":
                    run_data

            }), 502


        print(
            "PREDICTION ID:",
            prediction_id
        )


        # ====================================================
        # POLL RESULT
        # ====================================================

        for attempt in range(60):

            time.sleep(3)


            status_response = requests.get(

                FASHN_URL
                + "/status/"
                + prediction_id,

                headers={

                    "Authorization":
                        "Bearer " + FASHN_API_KEY

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
                "STATUS:",
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

                        "success": False,

                        "error":
                            "FASHN completed but returned no image.",

                        "prediction_id":
                            prediction_id

                    }), 502


                print("")
                print("========================================")
                print("✅ VTON COMPLETE")
                print("OUTPUT COUNT:", len(output))
                print("========================================")


                return jsonify({

                    "success": True,

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
                    "FASHN GENERATION FAILED:",
                    status_data
                )


                return jsonify({

                    "success": False,

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


            # ------------------------------------------------
            # UNKNOWN STATUS
            # ------------------------------------------------

            if status is None:

                print(
                    "WARNING: No status returned"
                )


        # ====================================================
        # POLLING TIMEOUT
        # ====================================================

        return jsonify({

            "success": False,

            "error":
                "FASHN prediction timed out.",

            "prediction_id":
                prediction_id

        }), 504


    # ========================================================
    # TIMEOUT
    # ========================================================

    except requests.exceptions.Timeout:

        print(
            "❌ FASHN REQUEST TIMEOUT"
        )


        return jsonify({

            "success": False,

            "error":
                "FASHN request timed out."

        }), 504


    # ========================================================
    # NETWORK ERROR
    # ========================================================

    except requests.exceptions.RequestException as e:

        print(
            "❌ FASHN NETWORK ERROR:",
            repr(e)
        )


        return jsonify({

            "success": False,

            "error":
                "Could not connect to FASHN.",

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

            "success": False,

            "error":
                str(e)

        }), 500


# ============================================================
# OPTIONAL OLD ROUTE
#
# Keep this so an old website endpoint doesn't break.
# It points to the same new VTON engine.
# ============================================================

@app.route(
    "/api/fashn-tryon",
    methods=["POST"]
)
def fashn_tryon():

    return tryon()


# ============================================================
# 404
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "success": False,

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
