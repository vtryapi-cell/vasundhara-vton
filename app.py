import os
from flask import Flask, jsonify

app = Flask(__name__)

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

@app.get("/routes")
def routes():
    return jsonify([
        str(rule)
        for rule in app.url_map.iter_rules()
    ])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
