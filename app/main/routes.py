import os
from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app, jsonify
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app import db
from app.database.models import Upload, Prediction
from app.ai.predict import run_inference
from app.ai.tcia_service import get_tcia_reference_cases, is_tcia_available
import uuid
from datetime import datetime
from app.ai.references import get_references
from config import Config

main_bp = Blueprint("main", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dcm", "nii", "gz"}

SCAN_TYPE_TO_MODALITY = {
    "brain_mri":  "Brain MRI",
    "chest_xray": "Chest X-Ray",
    "ct_scan":    "CT Scan",
    "mri_spine":  "Spine/Body MRI",
    "abdominal":  "Abdominal Scan",
    "skin":       "Dermatology / Skin",
    "xray_bone":  "X-Ray Bone & Joint",
    "general":    "General / Unknown",
}


def allowed_file(filename: str) -> bool:
    """
    Robust extension check that handles:
    - Normal files:          scan.dcm, image.png
    - Double extensions:     brain.nii.gz
    - Uppercase extensions:  SCAN.DCM
    - Windows full paths:    C:\\Users\\...\\scan.dcm
    """
    if not filename:
        return False
    filename = os.path.basename(filename)
    name_lower = filename.lower()
    if name_lower.endswith(".nii.gz"):
        return True
    if "." not in name_lower:
        return False
    ext = name_lower.rsplit(".", 1)[-1]
    return ext in ALLOWED_EXTENSIONS


def _remaining_scan_slots(user_id: int) -> tuple[int, int]:
    limit = Config.USER_SCAN_LIMIT
    used = Upload.query.filter_by(user_id=user_id).count()
    remaining = max(limit - used, 0)
    return remaining, limit


def detect_modality(filename: str, scan_type: str = "general") -> str:
    if scan_type and scan_type != "general":
        return SCAN_TYPE_TO_MODALITY.get(scan_type, "Unknown")
    ext = filename.rsplit(".", 1)[-1].lower()
    return {
        "dcm":  "CT/MRI (DICOM)",
        "nii":  "MRI (NIfTI)",
        "gz":   "MRI (NIfTI)",
        "png":  "Unknown",
        "jpg":  "Unknown",
        "jpeg": "Unknown",
    }.get(ext, "Unknown")


@main_bp.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    return redirect(url_for("auth.login"))


@main_bp.route("/dashboard")
@login_required
def dashboard():
    uploads = (
        Upload.query.filter_by(user_id=current_user.id)
        .order_by(Upload.upload_date.desc())
        .limit(10)
        .all()
    )
    from app.database.models import LabResult

    used_scan = Upload.query.filter_by(user_id=current_user.id).count()
    used_lab = LabResult.query.filter_by(user_id=current_user.id).count()
    scan_limit = Config.USER_SCAN_LIMIT
    lab_limit = Config.USER_LAB_SCAN_LIMIT

    stats = {
        "total": used_scan,
        "high_priority": db.session.query(Prediction)
            .join(Upload)
            .filter(Upload.user_id == current_user.id, Prediction.priority == "HIGH")
            .count(),
        "analyzed": db.session.query(Prediction)
            .join(Upload)
            .filter(Upload.user_id == current_user.id)
            .count(),
        "scan_limit": scan_limit,
        "scan_used": used_scan,
        "scan_remaining": max(scan_limit - used_scan, 0),
        "lab_limit": lab_limit,
        "lab_used": used_lab,
        "lab_remaining": max(lab_limit - used_lab, 0),
    }
    return render_template("dashboard.html", uploads=uploads, stats=stats)


@main_bp.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        remaining, limit = _remaining_scan_slots(current_user.id)
        if remaining <= 0:
            flash(
                f"You reached your scan limit ({limit}). Please contact admin to increase your plan.",
                "warning",
            )
            return redirect(request.url)

        if "file" not in request.files:
            flash("No file part in request.", "danger")
            return redirect(request.url)

        file = request.files["file"]
        if not file or file.filename == "":
            flash("No file selected.", "danger")
            return redirect(request.url)

        original_name = os.path.basename(file.filename)

        current_app.logger.info(
            f"Upload attempt: filename='{original_name}' "
            f"allowed={allowed_file(original_name)}"
        )

        if not allowed_file(original_name):
            flash(
                f"Unsupported file type '{original_name.rsplit('.', 1)[-1] if '.' in original_name else 'unknown'}'. "
                "Supported: PNG, JPG, DCM, NII, NII.GZ",
                "danger"
            )
            return redirect(request.url)

        scan_type = request.form.get("scan_type", "general")

        # FIX: Correct extension extraction — handles .nii.gz double extension
        name_lower = original_name.lower()
        if name_lower.endswith(".nii.gz"):
            ext = "nii.gz"
        else:
            ext = name_lower.rsplit(".", 1)[-1]

        unique_name = f"{uuid.uuid4().hex}.{ext}"
        save_dir    = os.path.join(current_app.config["UPLOAD_FOLDER"], str(current_user.id))
        os.makedirs(save_dir, exist_ok=True)
        file_path   = os.path.join(save_dir, unique_name)

        try:
            file.save(file_path)
            current_app.logger.info(
                f"File saved: {file_path} ({os.path.getsize(file_path)} bytes)"
            )
        except Exception as save_exc:
            current_app.logger.error(f"File save failed: {save_exc}")
            flash(f"Could not save file: {save_exc}", "danger")
            return redirect(request.url)

        # "nii.gz" → "nii" for the DB file_type field
        db_file_type = ext.split(".")[0] if "." in ext else ext

        upload_record = Upload(
            user_id=current_user.id,
            filename=unique_name,
            original_name=original_name,
            file_path=file_path,
            file_type=db_file_type,
            modality=detect_modality(original_name, scan_type),
        )
        db.session.add(upload_record)
        db.session.commit()

        try:
            prediction = run_inference(
                file_path,
                upload_record.id,
                ext,
                scan_type=scan_type,
            )
            db.session.add(prediction)
            db.session.commit()
            flash("Analysis complete!", "success")
            return redirect(url_for("main.result", upload_id=upload_record.id))
        except Exception as exc:
            current_app.logger.error(f"Inference error: {exc}")
            flash(f"File uploaded but analysis failed: {exc}", "warning")
            return redirect(url_for("main.dashboard"))

    return render_template("upload.html")


@main_bp.route("/result/<int:upload_id>")
@login_required
def result(upload_id):
    upload_record = Upload.query.filter_by(id=upload_id, user_id=current_user.id).first_or_404()
    prediction    = upload_record.prediction

    tcia_data = None
    if prediction and prediction.result_label:
        modality_to_scan = {v: k for k, v in SCAN_TYPE_TO_MODALITY.items()}
        scan_type = modality_to_scan.get(upload_record.modality, "general")
        try:
            tcia_data = get_tcia_reference_cases(
                cancer_label=prediction.result_label,
                scan_type=scan_type,
                max_results=3,
            )
        except Exception as exc:
            current_app.logger.warning(f"TCIA lookup failed (non-critical): {exc}")
            tcia_data = None

    clinical_refs = get_references(prediction.result_label, max_refs=4) if prediction else []

    return render_template(
        "result.html",
        upload=upload_record,
        prediction=prediction,
        tcia_data=tcia_data,
        clinical_refs=clinical_refs,
    )


@main_bp.route("/history")
@login_required
def history():
    page = request.args.get("page", 1, type=int)
    pagination = (
        Upload.query.filter_by(user_id=current_user.id)
        .order_by(Upload.upload_date.desc())
        .paginate(page=page, per_page=12, error_out=False)
    )
    return render_template("history.html", pagination=pagination)


@main_bp.route("/delete/<int:upload_id>", methods=["POST"])
@login_required
def delete_upload(upload_id):
    upload_record = Upload.query.filter_by(id=upload_id, user_id=current_user.id).first_or_404()
    try:
        if os.path.exists(upload_record.file_path):
            os.remove(upload_record.file_path)
        db.session.delete(upload_record)
        db.session.commit()
        flash("Scan deleted.", "success")
    except Exception as e:
        flash(f"Error deleting scan: {e}", "danger")
    return redirect(url_for("main.history"))


@main_bp.route("/api/upload-status/<int:upload_id>")
@login_required
def upload_status(upload_id):
    upload_record = Upload.query.filter_by(id=upload_id, user_id=current_user.id).first()
    if not upload_record:
        return jsonify({"status": "not_found"}), 404
    pred = upload_record.prediction
    return jsonify({
        "status":     "complete" if pred else "processing",
        "label":      pred.result_label if pred else None,
        "confidence": pred.confidence_score if pred else None,
        "priority":   pred.priority if pred else None,
    })


@main_bp.route("/api/tcia-status")
@login_required
def tcia_status():
    available = is_tcia_available()
    return jsonify({
        "tcia_available": available,
        "tcia_base_url":  "https://services.cancerimagingarchive.net/nbia-api/services/v1",
    })