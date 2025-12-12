"""
Lightweight Flask server to serve the static UI and an API endpoint that runs the pipeline.

Run: python app.py
Then open: http://localhost:8000
"""

from __future__ import annotations

import json
from pathlib import Path

import os
from flask import Flask, jsonify, send_from_directory

from pipeline import run_pipeline

app = Flask(__name__, static_folder="web", static_url_path="")


@app.route("/api/run", methods=["POST"])
def api_run():
    payload = run_pipeline(return_meta=True, include_graph=True)
    return jsonify(payload)


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>", methods=["GET"])
def static_proxy(path):
    full_path = Path(app.static_folder) / path
    if full_path.exists():
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
