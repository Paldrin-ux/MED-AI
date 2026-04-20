"""
Database models – Medical AI System (v2.1)

Changes from v2.0:
  LabResult table — added:
    • panels_json        : JSON array — one object per uploaded panel file
                           [{filename, panel_type, findings_json, normal, abnormal, critical}, ...]
    • combined_diagnosis : JSON array — cross-panel synthesised clinical diagnoses
                           [{condition, confidence, evidence, icd10}, ...]
    • panel_count        : how many files were uploaded in this session
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
    role          = db.Column(db.String(20), default="clinician")
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
    file_type     = db.Column(db.String(20))
    modality      = db.Column(db.String(20))
    upload_date   = db.Column(db.DateTime, default=datetime.utcnow)

    prediction = db.relationship(
        "Prediction", backref="upload", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Upload {self.filename}>"


# ---------------------------------------------------------------------------
# Prediction  (v2.0 — unchanged)
# ---------------------------------------------------------------------------

class Prediction(db.Model):
    __tablename__ = "predictions"

    id               = db.Column(db.Integer, primary_key=True)
    upload_id        = db.Column(db.Integer, db.ForeignKey("uploads.id"), nullable=False)
    result_label     = db.Column(db.String(120), nullable=False)
    confidence_score = db.Column(db.Float,       nullable=False)
    raw_scores       = db.Column(db.Text)
    priority         = db.Column(db.String(20))
    findings         = db.Column(db.Text)
    condition        = db.Column(db.String(200))
    severity         = db.Column(db.String(60))
    recommendation   = db.Column(db.Text)
    differential     = db.Column(db.Text)
    explanation      = db.Column(db.Text)
    heatmap_path     = db.Column(db.String(512))
    validation_notes = db.Column(db.Text)
    disclaimer       = db.Column(db.Text)
    analysis_date    = db.Column(db.DateTime, default=datetime.utcnow)
    model_version    = db.Column(db.String(100), default="gemini-vision-v2")

    @property
    def is_high_confidence(self):
        return (self.confidence_score or 0.0) >= 0.70

    @property
    def needs_differential(self):
        return (self.confidence_score or 0.0) < 0.70

    @property
    def was_rule_corrected(self):
        return bool(self.validation_notes and "Rule correction" in (self.validation_notes or ""))

    def to_dict(self):
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
# LabResult  (v2.1 — multi-panel support added)
# ---------------------------------------------------------------------------

class LabResult(db.Model):
    """
    Stores one lab analysis session which may contain 1–N uploaded panel files.

    New in v2.1
    -----------
    panel_count        — integer: how many files were submitted together
    panels_json        — JSON array, one element per uploaded file:
                         [
                           {
                             "filename":     "cbc.pdf",
                             "panel_type":   "CBC",          # auto-detected label
                             "normal":       14,
                             "abnormal":     3,
                             "critical":     0,
                             "findings_json": [...]           # per-test rows
                           },
                           ...
                         ]
    combined_diagnosis — JSON array of cross-panel clinical impressions:
                         [
                           {
                             "condition":   "Type 2 Diabetes Mellitus",
                             "confidence":  "High",          # High | Moderate | Low
                             "evidence":    "Elevated HbA1c 8.2%, fasting glucose 142 mg/dL, ...",
                             "icd10":       "E11"
                           },
                           ...
                         ]
    """

    __tablename__ = "lab_results"

    # ── identity ────────────────────────────────────────────────────────────
    id                 = db.Column(db.Integer, primary_key=True)
    user_id            = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    # ── file metadata ────────────────────────────────────────────────────────
    original_filename  = db.Column(db.String(260))   # first file name (or comma-joined for multi)
    file_type          = db.Column(db.String(20))
    panel_count        = db.Column(db.Integer, default=1)   # NEW v2.1

    # ── patient demographics ─────────────────────────────────────────────────
    patient_name       = db.Column(db.String(200))
    patient_age        = db.Column(db.String(50))
    patient_id_ref     = db.Column(db.String(100))
    test_date          = db.Column(db.String(100))
    ordering_physician = db.Column(db.String(200))
    facility           = db.Column(db.String(200))

    # ── aggregate narrative ──────────────────────────────────────────────────
    summary            = db.Column(db.Text)
    key_concerns       = db.Column(db.Text)
    interpretation     = db.Column(db.Text)
    next_steps         = db.Column(db.Text)
    urgency            = db.Column(db.String(20))
    disclaimer         = db.Column(db.Text)

    # ── aggregate counts (totals across all panels) ──────────────────────────
    normal_count       = db.Column(db.Integer, default=0)
    abnormal_count     = db.Column(db.Integer, default=0)
    critical_count     = db.Column(db.Integer, default=0)

    # ── structured data ──────────────────────────────────────────────────────
    findings_json      = db.Column(db.Text)   # flat list — all tests from all panels combined
    panels_json        = db.Column(db.Text)   # NEW v2.1 — per-panel breakdown (see docstring)
    combined_diagnosis = db.Column(db.Text)   # NEW v2.1 — cross-panel clinical impressions

    # ── engine metadata ──────────────────────────────────────────────────────
    model_used         = db.Column(db.String(100))
    analysis_date      = db.Column(db.DateTime, default=datetime.utcnow)

    # ── convenience ─────────────────────────────────────────────────────────

    @property
    def is_multi_panel(self):
        return (self.panel_count or 1) > 1

    def __repr__(self):
        return f"<LabResult #{self.id} {self.patient_name} panels={self.panel_count}>"