"""
Lab Results Analysis Route — with persistent history + delete
Handles PDF/DOC/image lab result uploads, AI analysis via Gemini Vision,
saves results to the database, and provides history + delete.
"""

import os
import json
import base64
import logging
import tempfile
import urllib.request
import urllib.error
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Blueprint, render_template, redirect, url_for, flash, request, abort
from flask_login import login_required, current_user

logger = logging.getLogger(__name__)

lab_bp = Blueprint("lab", __name__)

ALLOWED_LAB_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "docx", "doc", "txt", "csv"}
GEMINI_MODELS   = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def allowed_lab_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_LAB_EXTENSIONS


def encode_file_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_mime_type(ext):
    return {
        "pdf":  "application/pdf",
        "png":  "image/png",
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "doc":  "application/msword",
        "txt":  "text/plain",
        "csv":  "text/csv",
    }.get(ext.lower(), "application/octet-stream")


LAB_ANALYSIS_PROMPT = """You are a highly experienced medical laboratory specialist and clinical physician.
Analyze the provided lab result document carefully and generate a comprehensive, patient-friendly report.

Your report MUST follow EXACTLY this format — use these exact field names:

PATIENT_NAME: <full name if visible, else "Not specified">
PATIENT_AGE: <age if visible, else "Not specified">
PATIENT_ID: <ID/MRN if visible, else "Not specified">
TEST_DATE: <date of test if visible, else "Not specified">
ORDERING_PHYSICIAN: <doctor name if visible, else "Not specified">
FACILITY: <lab/hospital name if visible, else "Not specified">

SUMMARY: <2-3 sentence plain English summary of the overall findings>

ABNORMAL_COUNT: <number of abnormal values found>
CRITICAL_COUNT: <number of critical/panic values found>
NORMAL_COUNT: <number of normal values found>

FINDINGS_JSON: <a JSON array of finding objects, each with keys: test_name, value, unit, reference_range, status (Normal/Low/High/Critical), clinical_significance>
Example: [{"test_name":"Sodium","value":"127.95","unit":"mmol/L","reference_range":"137-145","status":"Low","clinical_significance":"Hyponatremia — may cause fatigue, confusion in elderly"}]

KEY_CONCERNS: <numbered list of the most important abnormal findings and why they matter>

INTERPRETATION: <3-5 sentences: clinical interpretation, possible underlying conditions, age/context>

NEXT_STEPS: <specific recommended next steps: follow-up tests, specialist referrals, urgency>

URGENCY: <ROUTINE or SOON or URGENT or EMERGENCY>

DISCLAIMER: This analysis is for informational purposes only and does not constitute a medical diagnosis. Please consult your physician to interpret these results in the context of your overall health.

Be thorough, accurate, and compassionate. Use plain language patients can understand."""


def call_gemini_lab_analysis(file_path: str, ext: str) -> dict:
    api_keys = [
        os.environ.get("GEMINI_API_KEY",   ""),
        os.environ.get("GEMINI_API_KEY_2", ""),
        os.environ.get("GEMINI_API_KEY_3", ""),
    ]
    api_keys = [k for k in api_keys if k]

    if not api_keys:
        raise RuntimeError("No GEMINI_API_KEY set")

    mime_type = get_mime_type(ext)

    if ext.lower() in ("txt", "csv"):
        with open(file_path, "r", errors="replace") as f:
            text_content = f.read()
        parts = [{"text": f"Lab result document:\n\n{text_content}\n\n{LAB_ANALYSIS_PROMPT}"}]
    else:
        file_b64 = encode_file_base64(file_path)
        parts = [
            {"inline_data": {"mime_type": mime_type, "data": file_b64}},
            {"text": LAB_ANALYSIS_PROMPT},
        ]

    payload = json.dumps({
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096},
    }).encode("utf-8")

    last_error = "No keys tried"

    for api_key in api_keys:
        for model in GEMINI_MODELS:
            url = f"{GEMINI_API_BASE}/{model}:generateContent?key={api_key}"
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                text = raw["candidates"][0]["content"]["parts"][0]["text"].strip()
                return parse_lab_response(text, model)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8")
                last_error = f"{model} HTTP {e.code}: {body[:300]}"
                if e.code in (429, 404):
                    continue
                raise RuntimeError(last_error)
            except Exception as e:
                last_error = f"{model}: {e}"
                continue

    raise RuntimeError(f"All Gemini keys/models exhausted. Last: {last_error}")


def parse_lab_response(text: str, model: str) -> dict:
    def get_field(key, default=""):
        for line in text.splitlines():
            s = line.strip()
            if s.upper().startswith(key.upper() + ":"):
                val = s[len(key)+1:].strip()
                if val:
                    return val
        return default

    def get_multiline(key):
        next_keys = [
            "PATIENT_NAME","PATIENT_AGE","PATIENT_ID","TEST_DATE",
            "ORDERING_PHYSICIAN","FACILITY","SUMMARY","ABNORMAL_COUNT",
            "CRITICAL_COUNT","NORMAL_COUNT","FINDINGS_JSON","KEY_CONCERNS",
            "INTERPRETATION","NEXT_STEPS","URGENCY","DISCLAIMER",
        ]
        lines = text.splitlines()
        collecting = False
        result = []
        for line in lines:
            s = line.strip()
            if s.upper().startswith(key.upper() + ":"):
                val = s[len(key)+1:].strip()
                if val:
                    result.append(val)
                collecting = True
                continue
            if collecting:
                if any(s.upper().startswith(k + ":") for k in next_keys if k != key):
                    break
                if s:
                    result.append(s)
        return " ".join(result).strip()

    findings = []
    findings_raw = get_multiline("FINDINGS_JSON")
    try:
        start = findings_raw.find("[")
        end   = findings_raw.rfind("]")
        if start != -1 and end != -1:
            findings = json.loads(findings_raw[start:end+1])
    except Exception:
        findings = []

    try:    abnormal_count = int(get_field("ABNORMAL_COUNT", "0"))
    except: abnormal_count = 0
    try:    critical_count = int(get_field("CRITICAL_COUNT", "0"))
    except: critical_count = 0
    try:    normal_count   = int(get_field("NORMAL_COUNT",   "0"))
    except: normal_count   = 0

    urgency = get_field("URGENCY", "ROUTINE").upper()
    if urgency not in ("ROUTINE", "SOON", "URGENT", "EMERGENCY"):
        urgency = "ROUTINE"

    return {
        "patient_name":       get_field("PATIENT_NAME",       "Not specified"),
        "patient_age":        get_field("PATIENT_AGE",        "Not specified"),
        "patient_id":         get_field("PATIENT_ID",         "Not specified"),
        "test_date":          get_field("TEST_DATE",          "Not specified"),
        "ordering_physician": get_field("ORDERING_PHYSICIAN", "Not specified"),
        "facility":           get_field("FACILITY",           "Not specified"),
        "summary":            get_multiline("SUMMARY"),
        "abnormal_count":     abnormal_count,
        "critical_count":     critical_count,
        "normal_count":       normal_count,
        "findings":           findings,
        "findings_json":      json.dumps(findings),
        "key_concerns":       get_multiline("KEY_CONCERNS"),
        "interpretation":     get_multiline("INTERPRETATION"),
        "next_steps":         get_multiline("NEXT_STEPS"),
        "urgency":            urgency,
        "disclaimer":         get_multiline("DISCLAIMER"),
        "model_used":         model,
    }


@lab_bp.route("/lab-results", methods=["GET", "POST"])
@login_required
def lab_upload():
    if request.method == "POST":
        if "lab_file" not in request.files:
            flash("No file selected.", "danger")
            return redirect(request.url)

        file = request.files["lab_file"]
        if not file or file.filename == "":
            flash("No file selected.", "danger")
            return redirect(request.url)

        if not allowed_lab_file(file.filename):
            flash("Unsupported file type. Upload PDF, PNG, JPG, DOCX, or TXT.", "danger")
            return redirect(request.url)

        ext           = file.filename.rsplit(".", 1)[1].lower()
        original_name = secure_filename(file.filename)

        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            api_key = (
                os.environ.get("GEMINI_API_KEY") or
                os.environ.get("GEMINI_API_KEY_2") or
                os.environ.get("GEMINI_API_KEY_3")
            )
            if not api_key:
                flash("No GEMINI_API_KEY configured.", "danger")
                return redirect(request.url)

            analysis = call_gemini_lab_analysis(tmp_path, ext)

            from app.database.models import LabResult
            from app import db

            lab_record = LabResult(
                user_id            = current_user.id,
                original_filename  = original_name,
                file_type          = ext,
                patient_name       = analysis["patient_name"],
                patient_age        = analysis["patient_age"],
                patient_id_ref     = analysis["patient_id"],
                test_date          = analysis["test_date"],
                ordering_physician = analysis["ordering_physician"],
                facility           = analysis["facility"],
                summary            = analysis["summary"],
                key_concerns       = analysis["key_concerns"],
                interpretation     = analysis["interpretation"],
                next_steps         = analysis["next_steps"],
                urgency            = analysis["urgency"],
                disclaimer         = analysis["disclaimer"],
                normal_count       = analysis["normal_count"],
                abnormal_count     = analysis["abnormal_count"],
                critical_count     = analysis["critical_count"],
                findings_json      = analysis["findings_json"],
                model_used         = analysis["model_used"],
                analysis_date      = datetime.utcnow(),
            )
            db.session.add(lab_record)
            db.session.commit()

            return redirect(url_for("lab.lab_result_view", result_id=lab_record.id))

        except Exception as e:
            logger.error(f"Lab analysis failed: {e}")
            flash(f"Analysis failed: {str(e)[:200]}", "danger")
            return redirect(request.url)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return render_template("lab_upload.html")


@lab_bp.route("/lab-results/<int:result_id>")
@login_required
def lab_result_view(result_id):
    from app.database.models import LabResult
    record = LabResult.query.get_or_404(result_id)
    if record.user_id != current_user.id:
        abort(403)

    findings = []
    if record.findings_json:
        try:
            findings = json.loads(record.findings_json)
        except Exception:
            findings = []

    return render_template("lab_result.html", record=record, findings=findings)


@lab_bp.route("/lab-results/<int:result_id>/delete", methods=["POST"])
@login_required
def lab_delete(result_id):
    from app.database.models import LabResult
    from app import db
    record = LabResult.query.get_or_404(result_id)
    if record.user_id != current_user.id:
        abort(403)
    db.session.delete(record)
    db.session.commit()
    flash("Lab result deleted.", "info")
    return redirect(url_for("lab.lab_history"))


@lab_bp.route("/lab-history")
@login_required
def lab_history():
    from app.database.models import LabResult
    records = (
        LabResult.query
        .filter_by(user_id=current_user.id)
        .order_by(LabResult.analysis_date.desc())
        .all()
    )
    stats = {
        "total":    len(records),
        "urgent":   sum(1 for r in records if r.urgency in ("URGENT", "EMERGENCY")),
        "abnormal": sum(1 for r in records if r.abnormal_count and r.abnormal_count > 0),
    }
    return render_template("lab_history.html", records=records, stats=stats)