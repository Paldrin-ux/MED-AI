"""
Database models – Medical AI System (v2.0)

Changes from v1.0:
  Prediction table — added:
    • findings          : radiological findings narrative
    • condition         : specific disease name
    • severity          : Mild / Moderate / Severe / Critical / ...
    • recommendation    : first-line treatment plan
    • differential      : top-2 alternative diagnoses when confidence < 0.70
    • explanation       : visual features that drove the prediction (Grad-CAM correlation)
    • validation_notes  : rule-based correction log (e.g. AD vs HCP disambiguation)
    • disclaimer        : mandatory safety disclaimer text
    • model_version     : widened to 100 chars (was 50)
"""

from datetime import datetime
from flask_login import UserMixin
from app import db, login_manager


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------

class User(db.Model, UserMixin):
    __tablename__ = "users"

    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(120), nullable=False)
    email         = db.Column(db.String(180), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role          = db.Column(db.String(20), default="clinician")   # clinician | radiologist | admin
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    uploads = db.relationship(
        "Upload", backref="owner", lazy=True, cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<User {self.email}>"


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

class Upload(db.Model):
    __tablename__ = "uploads"

    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    filename      = db.Column(db.String(260), nullable=False)
    original_name = db.Column(db.String(260))
    file_path     = db.Column(db.String(512), nullable=False)
    file_type     = db.Column(db.String(20))    # png | jpg | dcm | nii
    modality      = db.Column(db.String(20))    # MRI | CT | X-Ray | Unknown
    upload_date   = db.Column(db.DateTime, default=datetime.utcnow)

    prediction = db.relationship(
        "Prediction", backref="upload", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Upload {self.filename}>"


# ---------------------------------------------------------------------------
# Prediction  (v2.0 — extended)
# ---------------------------------------------------------------------------

class Prediction(db.Model):
    """
    Stores the full AI inference result for one uploaded medical image.

    Structured output fields (v2.0)
    --------------------------------
    findings        — 4-5 sentence radiological narrative written by the model
    condition       — specific disease name (may differ from the label bucket)
    severity        — Mild | Moderate | Severe | Critical | Early | Advanced | Unable to determine
    recommendation  — first-line treatment with drug names, dosages, and follow-up steps
    differential    — top-2 alternatives when confidence_score < 0.70; else 'None'
    explanation     — which visual features (sulci, hippocampus, ventricles ...) drove the label;
                      used to correlate with the Grad-CAM overlay
    validation_notes— populated by NeuroimagingValidator when a rule correction fires
                      (e.g. "AD overridden to HCP because sulci are compressed")
    disclaimer      — mandatory safety text injected on every result
    """

    __tablename__ = "predictions"

    # ── identity ────────────────────────────────────────────────────────────
    id        = db.Column(db.Integer, primary_key=True)
    upload_id = db.Column(db.Integer, db.ForeignKey("uploads.id"), nullable=False)

    # ── core result ─────────────────────────────────────────────────────────
    result_label     = db.Column(db.String(120), nullable=False)
    confidence_score = db.Column(db.Float,       nullable=False)
    raw_scores       = db.Column(db.Text)                           # JSON {label: score, ...}
    priority         = db.Column(db.String(20))                     # HIGH | MEDIUM | LOW

    # ── structured narrative (v2.0) ─────────────────────────────────────────
    findings         = db.Column(db.Text)                           # radiological findings
    condition        = db.Column(db.String(200))                    # specific disease name
    severity         = db.Column(db.String(60))                     # Mild / Moderate / Severe ...
    recommendation   = db.Column(db.Text)                           # treatment plan

    # ── differential & explainability (v2.0) ────────────────────────────────
    differential     = db.Column(db.Text)                           # alt diagnoses if low confidence
    explanation      = db.Column(db.Text)                           # visual features => label
    heatmap_path     = db.Column(db.String(512))                    # Grad-CAM overlay image path

    # ── safety & audit (v2.0) ───────────────────────────────────────────────
    validation_notes = db.Column(db.Text)                           # rule-correction log
    disclaimer       = db.Column(db.Text)                           # mandatory medical disclaimer

    # ── metadata ────────────────────────────────────────────────────────────
    analysis_date    = db.Column(db.DateTime, default=datetime.utcnow)
    model_version    = db.Column(db.String(100), default="gemini-vision-v2")  # widened from 50

    # ── convenience properties ───────────────────────────────────────────────

    @property
    def is_high_confidence(self) -> bool:
        """True when confidence meets the primary acceptance threshold (>= 0.70)."""
        return (self.confidence_score or 0.0) >= 0.70

    @property
    def needs_differential(self) -> bool:
        """True when confidence is too low for a single definitive label (< 0.70)."""
        return (self.confidence_score or 0.0) < 0.70

    @property
    def was_rule_corrected(self) -> bool:
        """True when NeuroimagingValidator overrode the raw model label."""
        return bool(self.validation_notes and "Rule correction" in (self.validation_notes or ""))

    # ── serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a JSON-safe dict for API responses and audit logging."""
        return {
            "id":                 self.id,
            "upload_id":          self.upload_id,
            "result_label":       self.result_label,
            "confidence_score":   self.confidence_score,
            "priority":           self.priority,
            "findings":           self.findings,
            "condition":          self.condition,
            "severity":           self.severity,
            "recommendation":     self.recommendation,
            "differential":       self.differential,
            "explanation":        self.explanation,
            "heatmap_path":       self.heatmap_path,
            "validation_notes":   self.validation_notes,
            "disclaimer":         self.disclaimer,
            "model_version":      self.model_version,
            "analysis_date":      self.analysis_date.isoformat() if self.analysis_date else None,
            "is_high_confidence": self.is_high_confidence,
            "was_rule_corrected": self.was_rule_corrected,
        }

    def __repr__(self):
        return f"<Prediction {self.result_label} {self.confidence_score:.2f}>"


# ---------------------------------------------------------------------------
# LabResult  (unchanged from v1.0)
# ---------------------------------------------------------------------------

class LabResult(db.Model):
    __tablename__ = "lab_results"

    id                 = db.Column(db.Integer, primary_key=True)
    user_id            = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    original_filename  = db.Column(db.String(260))
    file_type          = db.Column(db.String(20))
    patient_name       = db.Column(db.String(200))
    patient_age        = db.Column(db.String(50))
    patient_id_ref     = db.Column(db.String(100))
    test_date          = db.Column(db.String(100))
    ordering_physician = db.Column(db.String(200))
    facility           = db.Column(db.String(200))
    summary            = db.Column(db.Text)
    key_concerns       = db.Column(db.Text)
    interpretation     = db.Column(db.Text)
    next_steps         = db.Column(db.Text)
    urgency            = db.Column(db.String(20))
    disclaimer         = db.Column(db.Text)
    normal_count       = db.Column(db.Integer, default=0)
    abnormal_count     = db.Column(db.Integer, default=0)
    critical_count     = db.Column(db.Integer, default=0)
    findings_json      = db.Column(db.Text)
    model_used         = db.Column(db.String(100))
    analysis_date      = db.Column(db.DateTime, default=datetime.utcnow)