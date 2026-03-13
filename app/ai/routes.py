import os
from flask import Blueprint, jsonify
from flask_login import login_required
from app.ai.predict import MODALITY_PROMPTS

ai_bp = Blueprint("ai", __name__)


@ai_bp.route("/models")
@login_required
def models():
    available = [
        {
            "name": key,
            "description": val["context"],
            "classes": val["classes"],
        }
        for key, val in MODALITY_PROMPTS.items()
    ]
    return jsonify({"models": available})


@ai_bp.route("/health")
def health():
    api_key = os.environ.get("KIMI_API_KEY", "")
    key_set = bool(api_key)

    return jsonify({
        "status":            "ok" if key_set else "degraded",
        "kimi_key_set":      key_set,
        "inference_mode":    "Kimi K2 Vision (NVIDIA NIM)" if key_set else "simulation",
        "supported_formats": ["png", "jpg", "jpeg", "dcm", "nii"],
        "message": (
            "System ready for medical image analysis!"
            if key_set else
            "Set KIMI_API_KEY in .env to enable real AI analysis."
        ),
    })