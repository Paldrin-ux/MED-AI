from datetime import datetime
from flask_login import UserMixin
from app import db, login_manager


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(180), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default="clinician")  # clinician | radiologist | admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    uploads = db.relationship("Upload", backref="owner", lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.email}>"


class Upload(db.Model):
    __tablename__ = "uploads"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    filename = db.Column(db.String(260), nullable=False)
    original_name = db.Column(db.String(260))
    file_path = db.Column(db.String(512), nullable=False)
    file_type = db.Column(db.String(20))         # png | jpg | dcm | nii
    modality = db.Column(db.String(20))          # MRI | CT | X-Ray | Unknown
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    prediction = db.relationship("Prediction", backref="upload", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Upload {self.filename}>"


class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    upload_id = db.Column(db.Integer, db.ForeignKey("uploads.id"), nullable=False)
    result_label = db.Column(db.String(120), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    raw_scores = db.Column(db.Text)             # JSON string of all class scores
    priority = db.Column(db.String(20))         # HIGH | MEDIUM | LOW
    heatmap_path = db.Column(db.String(512))    # Grad-CAM overlay path
    analysis_date = db.Column(db.DateTime, default=datetime.utcnow)
    model_version = db.Column(db.String(50), default="densenet121-v1.0")

    def __repr__(self):
        return f"<Prediction {self.result_label} {self.confidence_score:.2f}>"

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
           
