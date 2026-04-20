"""
Lab Results Analysis Route  —  v2.1  (multi-panel)
===================================================
Changes from v1.0:
  • Accepts up to 10 files per submission (lab_files[] field)
  • Each file is analysed individually as a named panel
  • A second Gemini call synthesises cross-panel clinical impressions
    (e.g. CBC + CMP + HbA1c → Type 2 Diabetes Mellitus)
  • Results stored in LabResult.panels_json and .combined_diagnosis
  • Backward-compatible: single-file uploads work unchanged
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
from config import Config

logger = logging.getLogger(__name__)

lab_bp = Blueprint("lab", __name__)

ALLOWED_LAB_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "docx", "doc", "txt", "csv"}
GEMINI_MODELS   = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
MAX_FILES       = 10   # maximum panels per session


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def allowed_lab_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_LAB_EXTENSIONS


def _remaining_lab_scan_slots(user_id: int) -> tuple[int, int]:
    from app.database.models import LabResult
    limit     = Config.USER_LAB_SCAN_LIMIT
    used      = LabResult.query.filter_by(user_id=user_id).count()
    remaining = max(limit - used, 0)
    return remaining, limit


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


def _detect_panel_type(filename: str) -> str:
    """
    Heuristic panel-type label derived from the uploaded filename.
    Used as a label in the UI (e.g. "CBC", "Lipid Panel").
    Gemini will confirm / override this in the per-panel response.
    """
    name = filename.lower()
    if any(k in name for k in ("cbc", "complete blood", "blood count", "hemoglobin", "hematology")):
        return "CBC"
    if any(k in name for k in ("cmp", "bmp", "chemistry", "metabolic", "electrolyte", "renal", "liver", "lft", "bun", "creatinine")):
        return "Chemistry Panel"
    if any(k in name for k in ("lipid", "cholesterol", "triglyceride", "hdl", "ldl")):
        return "Lipid Panel"
    if any(k in name for k in ("thyroid", "tsh", "t3", "t4", "ft4", "ft3")):
        return "Thyroid Panel"
    if any(k in name for k in ("urine", "urinalysis", "ua ", "ua_", "ua.")):
        return "Urinalysis"
    if any(k in name for k in ("hba1c", "a1c", "glycated", "glucose", "diabetes")):
        return "HbA1c / Glucose"
    if any(k in name for k in ("coag", "pt ", "inr", "ptt", "aptt", "fibrinogen")):
        return "Coagulation"
    if any(k in name for k in ("culture", "sensitivity", "micro", "bacteria")):
        return "Microbiology"
    if any(k in name for k in ("hormone", "testosterone", "estrogen", "cortisol", "dhea")):
        return "Hormone Panel"
    if any(k in name for k in ("vitamin", "vit d", "vit b", "folate", "ferritin", "iron")):
        return "Vitamins & Minerals"
    return "Lab Report"


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

def _build_panel_prompt(panel_label: str) -> str:
    """Per-file analysis prompt — identifies itself as a specific panel."""
    return f"""You are a highly experienced medical laboratory specialist and clinical physician.
Analyze the provided lab document. This document appears to be a: {panel_label}.

Your report MUST follow EXACTLY this format — use these exact field names:

PATIENT_NAME: <full name if visible, else "Not specified">
PATIENT_AGE: <age if visible, else "Not specified">
PATIENT_ID: <ID/MRN if visible, else "Not specified">
TEST_DATE: <date of test if visible, else "Not specified">
ORDERING_PHYSICIAN: <doctor name if visible, else "Not specified">
FACILITY: <lab/hospital name if visible, else "Not specified">
PANEL_TYPE: <actual panel type you detected, e.g. CBC, Lipid Panel, Urinalysis, Chemistry Panel, Thyroid Panel, HbA1c, Coagulation, etc.>

SUMMARY: <2-3 sentence plain English summary of findings in THIS panel only>

ABNORMAL_COUNT: <number of abnormal values in this panel>
CRITICAL_COUNT: <number of critical/panic values in this panel>
NORMAL_COUNT: <number of normal values in this panel>

FINDINGS_JSON: <JSON array — every test in this panel. Each object: test_name, value, unit, reference_range, status (Normal/Low/High/Critical), clinical_significance>
Example: [{{"test_name":"Glucose","value":"142","unit":"mg/dL","reference_range":"70-99","status":"High","clinical_significance":"Elevated fasting glucose — may indicate diabetes or prediabetes"}}]

KEY_CONCERNS: <numbered list of abnormal findings from THIS panel that are clinically important>

URGENCY: <ROUTINE or SOON or URGENT or EMERGENCY — for THIS panel only>

Be thorough and accurate. Use plain language."""


SYNTHESIS_PROMPT = """You are a senior clinician with expertise in interpreting multi-panel laboratory results holistically.

You have been given structured JSON summaries of {n} laboratory panels from the same patient.

Panels provided:
{panels_summary}

Your task: synthesize ALL panels together to identify likely clinical diagnoses or conditions the patient may have.

Look for CROSS-PANEL PATTERNS, such as:
- High HbA1c + high fasting glucose + high triglycerides → Type 2 Diabetes with Dyslipidemia
- Low T4 + high TSH + high cholesterol → Hypothyroidism
- Low hemoglobin + low MCV + low ferritin → Iron-deficiency Anemia
- High creatinine + high BUN + low GFR → Chronic Kidney Disease
- High ALT + high AST + high bilirubin → Hepatic dysfunction
- High WBC + high neutrophils + high CRP → Active infection / inflammation
- High LDL + high total cholesterol + low HDL → Dyslipidemia / cardiovascular risk

COMBINED_DIAGNOSIS_JSON: <JSON array of clinical impressions. Each object must have:
  condition   — full clinical name (e.g. "Type 2 Diabetes Mellitus")
  confidence  — "High", "Moderate", or "Low"
  evidence    — 1-2 sentences citing the specific abnormal values that support this diagnosis
  icd10       — ICD-10 code (e.g. "E11")
  priority    — "HIGH", "MEDIUM", or "LOW"
>

OVERALL_SUMMARY: <3-5 sentence plain English summary that integrates ALL panel results into a coherent clinical picture for the patient>

OVERALL_KEY_CONCERNS: <numbered list of the most important findings across ALL panels>

OVERALL_INTERPRETATION: <4-6 sentences: integrated clinical interpretation, likely conditions, how panels relate to each other>

OVERALL_NEXT_STEPS: <specific next steps: tests to confirm diagnoses, specialist referrals, lifestyle changes, urgency>

OVERALL_URGENCY: <ROUTINE or SOON or URGENT or EMERGENCY — highest urgency across all panels>

DISCLAIMER: This analysis is for informational purposes only and does not constitute a medical diagnosis. Please consult your physician to interpret these results in the context of your overall health.

Respond ONLY with the fields above. Be clinically precise and patient-friendly."""


# ─────────────────────────────────────────────────────────────────────────────
# Gemini API calls
# ─────────────────────────────────────────────────────────────────────────────

def _get_api_keys() -> list[str]:
    keys = [
        os.environ.get("GEMINI_API_KEY",   ""),
        os.environ.get("GEMINI_API_KEY_2", ""),
        os.environ.get("GEMINI_API_KEY_3", ""),
    ]
    return [k for k in keys if k]


def _call_gemini(payload_bytes: bytes) -> str:
    """
    Try all API keys × all models. Return the raw text response.
    Raises RuntimeError if everything fails.
    """
    last_error = "No keys available"
    for api_key in _get_api_keys():
        for model in GEMINI_MODELS:
            url = f"{GEMINI_API_BASE}/{model}:generateContent?key={api_key}"
            req = urllib.request.Request(
                url, data=payload_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    raw  = json.loads(resp.read().decode("utf-8"))
                text = raw["candidates"][0]["content"]["parts"][0]["text"].strip()
                logger.info(f"Gemini response via {model} ({len(text)} chars)")
                return text
            except urllib.error.HTTPError as e:
                body       = e.read().decode("utf-8")
                last_error = f"{model} HTTP {e.code}: {body[:300]}"
                if e.code in (429, 404):
                    continue
                raise RuntimeError(last_error)
            except Exception as e:
                last_error = f"{model}: {e}"
                continue
    raise RuntimeError(f"All Gemini keys/models exhausted. Last: {last_error}")


def _analyse_single_panel(file_path: str, ext: str, panel_label: str) -> dict:
    """Call Gemini on one file; return parsed panel dict."""
    prompt    = _build_panel_prompt(panel_label)
    mime_type = get_mime_type(ext)

    if ext.lower() in ("txt", "csv"):
        with open(file_path, "r", errors="replace") as f:
            text_content = f.read()
        parts = [{"text": f"Lab result document:\n\n{text_content}\n\n{prompt}"}]
    else:
        file_b64 = encode_file_base64(file_path)
        parts    = [
            {"inline_data": {"mime_type": mime_type, "data": file_b64}},
            {"text": prompt},
        ]

    payload = json.dumps({
        "contents":       [{"parts": parts}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096},
    }).encode("utf-8")

    text = _call_gemini(payload)
    return _parse_panel_response(text)


def _synthesise_panels(panels: list[dict]) -> dict:
    """
    Second Gemini call: feed all panel summaries as JSON,
    ask for cross-panel diagnosis.
    """
    # Build a compact text summary of each panel for the prompt
    panel_summaries = []
    for i, p in enumerate(panels, 1):
        findings_preview = p.get("findings", [])[:20]  # cap to avoid token overflow
        panel_summaries.append({
            "panel_number": i,
            "panel_type":   p.get("panel_type", "Lab Report"),
            "filename":     p.get("filename", ""),
            "abnormal":     p.get("abnormal_count", 0),
            "critical":     p.get("critical_count", 0),
            "summary":      p.get("summary", ""),
            "key_concerns": p.get("key_concerns", ""),
            "findings":     findings_preview,
        })

    filled_prompt = SYNTHESIS_PROMPT.format(
        n=len(panels),
        panels_summary=json.dumps(panel_summaries, indent=2),
    )

    payload = json.dumps({
        "contents":       [{"parts": [{"text": filled_prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096},
    }).encode("utf-8")

    text = _call_gemini(payload)
    return _parse_synthesis_response(text)


# ─────────────────────────────────────────────────────────────────────────────
# Response parsers
# ─────────────────────────────────────────────────────────────────────────────

def _get_field(text: str, key: str, default: str = "") -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith(key.upper() + ":"):
            val = s[len(key) + 1:].strip()
            if val:
                return val
    return default


def _get_multiline(text: str, key: str) -> str:
    """Collect a multi-line field value until the next known field header."""
    next_keys = [
        "PATIENT_NAME", "PATIENT_AGE", "PATIENT_ID", "TEST_DATE",
        "ORDERING_PHYSICIAN", "FACILITY", "PANEL_TYPE", "SUMMARY",
        "ABNORMAL_COUNT", "CRITICAL_COUNT", "NORMAL_COUNT", "FINDINGS_JSON",
        "KEY_CONCERNS", "INTERPRETATION", "NEXT_STEPS", "URGENCY", "DISCLAIMER",
        "COMBINED_DIAGNOSIS_JSON", "OVERALL_SUMMARY", "OVERALL_KEY_CONCERNS",
        "OVERALL_INTERPRETATION", "OVERALL_NEXT_STEPS", "OVERALL_URGENCY",
    ]
    lines      = text.splitlines()
    collecting = False
    result     = []
    for line in lines:
        s = line.strip()
        if s.upper().startswith(key.upper() + ":"):
            val = s[len(key) + 1:].strip()
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


def _parse_json_array(raw: str) -> list:
    try:
        start = raw.find("[")
        end   = raw.rfind("]")
        if start != -1 and end != -1:
            return json.loads(raw[start:end + 1])
    except Exception:
        pass
    return []


def _parse_panel_response(text: str) -> dict:
    findings_raw = _get_multiline(text, "FINDINGS_JSON")
    findings     = _parse_json_array(findings_raw)

    def safe_int(key, default=0):
        try:    return int(_get_field(text, key, str(default)))
        except: return default

    urgency = _get_field(text, "URGENCY", "ROUTINE").upper()
    if urgency not in ("ROUTINE", "SOON", "URGENT", "EMERGENCY"):
        urgency = "ROUTINE"

    return {
        "patient_name":       _get_field(text, "PATIENT_NAME",       "Not specified"),
        "patient_age":        _get_field(text, "PATIENT_AGE",        "Not specified"),
        "patient_id":         _get_field(text, "PATIENT_ID",         "Not specified"),
        "test_date":          _get_field(text, "TEST_DATE",          "Not specified"),
        "ordering_physician": _get_field(text, "ORDERING_PHYSICIAN", "Not specified"),
        "facility":           _get_field(text, "FACILITY",           "Not specified"),
        "panel_type":         _get_field(text, "PANEL_TYPE",         "Lab Report"),
        "summary":            _get_multiline(text, "SUMMARY"),
        "key_concerns":       _get_multiline(text, "KEY_CONCERNS"),
        "abnormal_count":     safe_int("ABNORMAL_COUNT"),
        "critical_count":     safe_int("CRITICAL_COUNT"),
        "normal_count":       safe_int("NORMAL_COUNT"),
        "findings":           findings,
        "findings_json":      json.dumps(findings),
        "urgency":            urgency,
    }


def _parse_synthesis_response(text: str) -> dict:
    diag_raw = _get_multiline(text, "COMBINED_DIAGNOSIS_JSON")
    diagnoses = _parse_json_array(diag_raw)

    urgency = _get_field(text, "OVERALL_URGENCY", "ROUTINE").upper()
    if urgency not in ("ROUTINE", "SOON", "URGENT", "EMERGENCY"):
        urgency = "ROUTINE"

    return {
        "combined_diagnosis":  diagnoses,
        "summary":             _get_multiline(text, "OVERALL_SUMMARY"),
        "key_concerns":        _get_multiline(text, "OVERALL_KEY_CONCERNS"),
        "interpretation":      _get_multiline(text, "OVERALL_INTERPRETATION"),
        "next_steps":          _get_multiline(text, "OVERALL_NEXT_STEPS"),
        "urgency":             urgency,
        "disclaimer":          _get_multiline(text, "DISCLAIMER"),
    }


def _merge_patient_info(panels: list[dict]) -> dict:
    """
    Pick patient demographics from the first panel that has them.
    Prefer values that are not "Not specified".
    """
    fields = ["patient_name", "patient_age", "patient_id",
              "test_date", "ordering_physician", "facility"]
    merged = {f: "Not specified" for f in fields}
    for panel in panels:
        for f in fields:
            if merged[f] in ("Not specified", "", None):
                merged[f] = panel.get(f, "Not specified") or "Not specified"
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@lab_bp.route("/lab-results", methods=["GET", "POST"])
@login_required
def lab_upload():
    if request.method == "POST":
        remaining, limit = _remaining_lab_scan_slots(current_user.id)
        if remaining <= 0:
            flash(
                f"You reached your lab scan limit ({limit}). Please contact admin to increase your plan.",
                "warning",
            )
            return redirect(request.url)

        # Accept either the old single-file field ("lab_file") or the new
        # multi-file field ("lab_files[]") — keeps backward compatibility.
        files = request.files.getlist("lab_files[]")
        if not files or all(f.filename == "" for f in files):
            # fall back to legacy single-file field
            single = request.files.get("lab_file")
            files  = [single] if single and single.filename else []

        if not files:
            flash("No files selected.", "danger")
            return redirect(request.url)

        # Validate & cap
        files = [f for f in files if f and f.filename and allowed_lab_file(f.filename)]
        if not files:
            flash("No supported files found. Upload PDF, PNG, JPG, DOCX, TXT, or CSV.", "danger")
            return redirect(request.url)
        if len(files) > MAX_FILES:
            flash(f"Maximum {MAX_FILES} files per session. Only the first {MAX_FILES} will be used.", "warning")
            files = files[:MAX_FILES]

        if not _get_api_keys():
            flash("No GEMINI_API_KEY configured.", "danger")
            return redirect(request.url)

        # ── Step 1: analyse each file individually ───────────────────────────
        tmp_paths    = []
        panel_results = []
        original_names = []

        try:
            for file in files:
                ext           = file.filename.rsplit(".", 1)[1].lower()
                original_name = secure_filename(file.filename)
                original_names.append(original_name)
                panel_label   = _detect_panel_type(original_name)

                with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
                    file.save(tmp.name)
                    tmp_path = tmp.name
                tmp_paths.append(tmp_path)

                logger.info(f"Analysing panel: {original_name} → {panel_label}")
                try:
                    panel_data = _analyse_single_panel(tmp_path, ext, panel_label)
                except Exception as exc:
                    logger.error(f"Panel analysis failed for {original_name}: {exc}")
                    # Insert a placeholder so the session still completes
                    panel_data = {
                        "patient_name": "Not specified", "patient_age": "Not specified",
                        "patient_id": "Not specified", "test_date": "Not specified",
                        "ordering_physician": "Not specified", "facility": "Not specified",
                        "panel_type": panel_label, "summary": f"Analysis failed: {str(exc)[:120]}",
                        "key_concerns": "", "abnormal_count": 0, "critical_count": 0,
                        "normal_count": 0, "findings": [], "findings_json": "[]",
                        "urgency": "ROUTINE",
                    }

                panel_data["filename"] = original_name
                panel_results.append(panel_data)

            # ── Step 2: synthesis (multi-panel only) ─────────────────────────
            if len(panel_results) > 1:
                logger.info(f"Running cross-panel synthesis for {len(panel_results)} panels")
                try:
                    synthesis = _synthesise_panels(panel_results)
                except Exception as exc:
                    logger.error(f"Synthesis failed: {exc}")
                    # Fall back: aggregate naively
                    synthesis = {
                        "combined_diagnosis": [],
                        "summary":            " | ".join(p["summary"] for p in panel_results if p.get("summary")),
                        "key_concerns":       "\n".join(p["key_concerns"] for p in panel_results if p.get("key_concerns")),
                        "interpretation":     "Cross-panel synthesis unavailable. See individual panel summaries.",
                        "next_steps":         "Please consult your physician for an integrated interpretation.",
                        "urgency":            max((p["urgency"] for p in panel_results),
                                                  key=lambda u: {"ROUTINE":0,"SOON":1,"URGENT":2,"EMERGENCY":3}.get(u,0)),
                        "disclaimer":         "This analysis is for informational purposes only.",
                    }
            else:
                # Single file — use panel data directly
                p = panel_results[0]
                synthesis = {
                    "combined_diagnosis": [],
                    "summary":            p.get("summary", ""),
                    "key_concerns":       p.get("key_concerns", ""),
                    "interpretation":     "",
                    "next_steps":         "",
                    "urgency":            p.get("urgency", "ROUTINE"),
                    "disclaimer":         "This analysis is for informational purposes only. "
                                          "Please consult your physician.",
                }

            # ── Step 3: aggregate counts & build panels_json ─────────────────
            total_normal   = sum(p.get("normal_count",   0) for p in panel_results)
            total_abnormal = sum(p.get("abnormal_count", 0) for p in panel_results)
            total_critical = sum(p.get("critical_count", 0) for p in panel_results)

            all_findings = []
            for p in panel_results:
                all_findings.extend(p.get("findings", []))

            panels_for_db = [
                {
                    "filename":     p.get("filename", ""),
                    "panel_type":   p.get("panel_type", "Lab Report"),
                    "normal":       p.get("normal_count",   0),
                    "abnormal":     p.get("abnormal_count", 0),
                    "critical":     p.get("critical_count", 0),
                    "summary":      p.get("summary",        ""),
                    "key_concerns": p.get("key_concerns",   ""),
                    "findings_json": p.get("findings",      []),
                }
                for p in panel_results
            ]

            patient_info = _merge_patient_info(panel_results)

            # ── Step 4: first model name used ───────────────────────────────
            model_used = GEMINI_MODELS[0]  # best-effort; actual model logged above

            # ── Step 5: save to DB ───────────────────────────────────────────
            from app.database.models import LabResult
            from app import db

            lab_record = LabResult(
                user_id            = current_user.id,
                original_filename  = ", ".join(original_names),
                file_type          = files[0].filename.rsplit(".", 1)[1].lower() if files else "pdf",
                panel_count        = len(panel_results),
                patient_name       = patient_info["patient_name"],
                patient_age        = patient_info["patient_age"],
                patient_id_ref     = patient_info["patient_id"],
                test_date          = patient_info["test_date"],
                ordering_physician = patient_info["ordering_physician"],
                facility           = patient_info["facility"],
                summary            = synthesis["summary"],
                key_concerns       = synthesis["key_concerns"],
                interpretation     = synthesis.get("interpretation", ""),
                next_steps         = synthesis.get("next_steps", ""),
                urgency            = synthesis["urgency"],
                disclaimer         = synthesis.get("disclaimer", ""),
                normal_count       = total_normal,
                abnormal_count     = total_abnormal,
                critical_count     = total_critical,
                findings_json      = json.dumps(all_findings),
                panels_json        = json.dumps(panels_for_db),
                combined_diagnosis = json.dumps(synthesis.get("combined_diagnosis", [])),
                model_used         = model_used,
                analysis_date      = datetime.utcnow(),
            )
            db.session.add(lab_record)
            db.session.commit()

            return redirect(url_for("lab.lab_result_view", result_id=lab_record.id))

        except Exception as e:
            logger.error(f"Lab analysis session failed: {e}", exc_info=True)
            flash(f"Analysis failed: {str(e)[:200]}", "danger")
            return redirect(request.url)

        finally:
            for p in tmp_paths:
                try:
                    os.remove(p)
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

    # All findings (flat — for the combined table)
    findings = []
    if record.findings_json:
        try:
            findings = json.loads(record.findings_json)
        except Exception:
            findings = []

    # Per-panel breakdown
    panels = []
    if record.panels_json:
        try:
            panels = json.loads(record.panels_json)
        except Exception:
            panels = []

    # Cross-panel diagnoses
    combined_diagnosis = []
    if record.combined_diagnosis:
        try:
            combined_diagnosis = json.loads(record.combined_diagnosis)
        except Exception:
            combined_diagnosis = []

    return render_template(
        "lab_result.html",
        record=record,
        findings=findings,
        panels=panels,
        combined_diagnosis=combined_diagnosis,
    )


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