import os
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.get("/debug")
def debug():
    return "THIS IS THE NEW APP.PY"

@app.get("/")
def home():
    return "VASUNDHARA VTON IS LIVE"

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
        "routes": [str(rule) for rule in app.url_map.iter_rules()]
    }), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
