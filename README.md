# MedAI – AI Medical Image Analysis System

> **Disclaimer:** This system is for research and decision-support only.
> All AI predictions must be reviewed by a qualified radiologist before clinical use.

---

## Quick Start

```bash
# 1. Clone / unzip project
cd medical_ai_system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python app.py
# → Open http://localhost:5000
```

### Docker
```bash
docker compose up --build
# → http://localhost:5000
```

---

## Project Structure

```
medical_ai_system/
├── app.py                      ← Flask entry point
├── config.py                   ← Environment configs
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── app/
│   ├── __init__.py             ← App factory (create_app)
│   ├── auth/
│   │   └── routes.py           ← Register / Login / Logout
│   ├── main/
│   │   └── routes.py           ← Dashboard / Upload / Result / History
│   ├── ai/
│   │   ├── model_loader.py     ← ModelRegistry + DenseNet-121 loader
│   │   ├── preprocess.py       ← MONAI transform pipelines
│   │   ├── predict.py          ← Inference + Grad-CAM pipeline
│   │   ├── routes.py           ← /ai/health, /ai/models
│   │   └── weights/            ← Place .pt weight files here
│   ├── database/
│   │   └── models.py           ← User / Upload / Prediction ORM models
│   └── uploads/                ← Per-user scan storage
│
├── templates/
│   ├── base.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── upload.html
│   ├── result.html
│   └── history.html
│
└── static/
    ├── css/main.css
    └── js/{main,upload}.js
```

---

## Medical AI Architect Recommendations

### Pretrained Models for Brain Tumor Detection

#### Option 1 – MONAI Model Zoo (Recommended)
```python
# Download BraTS brain tumor segmentation bundle
from monai.bundle import download
download("brats_mri_segmentation", bundle_dir="app/ai/weights/")
# Then load with monai.bundle.ConfigParser
```

#### Option 2 – Kaggle Brain MRI Kaggle Dataset (Classification)
- Dataset: [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- Fine-tune MONAI DenseNet-121 (4 classes: No Tumor, Meningioma, Glioma, Pituitary)
- Save as `app/ai/weights/brain_tumor_densenet121.pt`

#### Option 3 – Hugging Face Hub
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="your-org/brain-tumor-densenet121",
    filename="model.pt"
)
```

### Pretrained Weights Placement
Place `.pt` files in `app/ai/weights/` with these exact filenames:
| File | Task |
|---|---|
| `brain_tumor_densenet121.pt` | Brain tumor (4-class) |
| `stroke_densenet121.pt` | Stroke detection |
| `chest_densenet121.pt` | Chest X-Ray pathology |

The system auto-detects and loads weights. If absent, it runs in **simulation mode** (random predictions for UI development).

---

## DICOM Preprocessing Pipeline (Best Practices)

```
Raw DICOM
  ↓ pydicom.dcmread()         — parse metadata + pixel array
  ↓ MONAI LoadImage(reader="PydicomReader")
  ↓ EnsureChannelFirst()      — (H,W) → (1,H,W)
  ↓ HU Windowing              — Brain: [-80,80] | Soft tissue: [-150,250]
  ↓ ScaleIntensity(0,1)       — Normalize to [0,1]
  ↓ Resize(224,224)           — Match model input
  ↓ NormalizeIntensity()      — Zero-mean unit-variance
  ↓ Repeat channel × 3       — Grayscale → RGB for ImageNet-pretrained backbone
  ↓ ToTensor()
  ↓ Unsqueeze(0)              — Add batch dim → (1,3,224,224)
```

### For 3-D NIfTI MRI (Volumetric)
- Use `SpatialPad + CropForeground` instead of Resize
- Take **middle axial slice** for 2-D inference (current implementation)
- For full 3-D: switch to MONAI SegResNet or 3-D DenseNet

---

## Efficient Inference Workflow

### Single-Image (current)
```
Request → Preprocess → Model.forward() → Softmax → Grad-CAM → Response
Latency: ~200ms CPU / ~30ms GPU (DenseNet-121)
```

### Batch / Async (production extension)
```python
# Use Celery + Redis for background inference
from celery import Celery
celery = Celery('medai', broker='redis://localhost:6379/0')

@celery.task
def async_infer(file_path, upload_id, ext):
    from app.ai.predict import run_inference
    ...
```

### GPU Scaling
```bash
# Set in docker-compose.yml or .env
INFERENCE_DEVICE=cuda
# Then use NVIDIA Docker runtime:
docker run --gpus all medai
```

---

## Security

| Feature | Implementation |
|---|---|
| Password hashing | `flask-bcrypt` (bcrypt, 12 rounds) |
| Session protection | `Flask-Login` + `SESSION_COOKIE_HTTPONLY` |
| File validation | Extension whitelist + MIME check |
| User isolation | Per-user upload directories |
| SQL injection | SQLAlchemy ORM parameterized queries |
| XSS | Jinja2 auto-escaping |

---

## Scalability Architecture

```
           ┌─────────────────┐
           │   Load Balancer │
           └────────┬────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
   ┌─────┴─────┐         ┌─────┴─────┐
   │ Flask App │         │ Flask App │   ← Gunicorn (stateless)
   └─────┬─────┘         └─────┬─────┘
         │                     │
   ┌─────┴─────────────────────┴─────┐
   │         Shared Storage          │   ← S3 / NFS for uploads
   └─────────────────────────────────┘
         │
   ┌─────┴──────┐     ┌─────────────┐
   │  Postgres  │     │ Redis Queue │   ← Replace SQLite for prod
   └────────────┘     └──────┬──────┘
                              │
                       ┌──────┴──────┐
                       │  Celery GPU │   ← Async inference workers
                       │   Workers   │
                       └─────────────┘
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | dev key | Flask secret – **change in production** |
| `FLASK_ENV` | `development` | `development` or `production` |
| `INFERENCE_DEVICE` | `cpu` | `cpu` or `cuda` |
| `DATABASE_URL` | SQLite | PostgreSQL URL for production |
| `PORT` | `5000` | Server port |
