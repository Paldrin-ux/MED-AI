"""
File serving routes – expose uploaded scans for browser preview.
Only serves files belonging to the authenticated user.
"""
import os
from flask import send_from_directory, abort
from flask_login import login_required, current_user
from app.main.routes import main_bp


@main_bp.route("/uploads/<int:user_id>/<path:filename>")
@login_required
def serve_upload(user_id, filename):
    # Security: users can only access their own files
    if user_id != current_user.id:
        abort(403)
    upload_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "app", "uploads", str(user_id)
    )
    return send_from_directory(upload_base, filename)
