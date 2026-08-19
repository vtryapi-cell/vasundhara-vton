import os
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

print("🔥🔥🔥 VASUNDHARA APP.PY LOADED 🔥🔥🔥")
print("APP FILE:", __file__)
print("CURRENT DIRECTORY:", os.getcwd())


@app.get("/")
def home():
    try:
        return render_template("index.html")
    except Exception:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vasundhara VTON</title>
        </head>
        <body>
            <h1>VASUNDHARA VTON IS LIVE</h1>
            <p>Flask server is running successfully.</p>

            <p>
                <a href="/health">Health Check</a>
            </p>

            <p>
                <a href="/debug">Debug</a>
            </p>

            <p>
                <a href="/test">Test</a>
            </p>
        </body>
        </html>
        """


@app.get("/debug")
def debug():
    return jsonify({
        "status": "OK",
        "message": "VASUNDHARA NEW APP.PY IS RUNNING",
        "app_file": __file__,
        "current_directory": os.getcwd(),
        "routes": [
            str(rule)
            for rule in app.url_map.iter_rules()
        ]
    })


@app.get("/test")
def test():
    return "TEST ROUTE WORKS"


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "message": "VASUNDHARA VTON SERVER IS RUNNING"
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Flask 404",
        "requested_path": request.path,
        "app_file": __file__,
        "routes": [
            str(rule)
            for rule in app.url_map.iter_rules()
        ]
    }), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))

    print("🚀 Starting VASUNDHARA VTON")
    print("🚀 PORT:", port)

    app.run(
        host="0.0.0.0",
        port=port
    )
