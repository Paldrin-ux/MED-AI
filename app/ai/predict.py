"""
Inference pipeline – Medical AI System
Uses Google Gemini Vision API for image analysis.
"""

import os
import json
import base64
import logging
import tempfile
import shutil
import urllib.request
import urllib.error
from datetime import datetime

from app.database.models import Prediction

logger = logging.getLogger(__name__)

VISION_SUPPORTED = {"png", "jpg", "jpeg", "dcm", "nii", "gz", "nii.gz"}

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODELS   = [
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash",
]


def _dicom_to_png(dcm_path: str) -> str:
    try:
        import pydicom
        import numpy as np
        from PIL import Image

        ds  = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype(float)

        if arr.ndim == 3 and arr.shape[0] > 3:
            arr = arr[arr.shape[0] // 2]

        try:
            intercept = float(ds.RescaleIntercept) if hasattr(ds, "RescaleIntercept") else 0
            slope     = float(ds.RescaleSlope)     if hasattr(ds, "RescaleSlope")     else 1
            arr = arr * slope + intercept
        except Exception:
            pass

        try:
            wc = float(ds.WindowCenter) if hasattr(ds, "WindowCenter") else None
            ww = float(ds.WindowWidth)  if hasattr(ds, "WindowWidth")  else None
            if isinstance(wc, list): wc = wc[0]
            if isinstance(ww, list): ww = ww[0]
            if wc and ww:
                arr = np.clip(arr, wc - ww / 2, wc + ww / 2)
        except Exception:
            pass

        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        arr = arr.astype(np.uint8)

        img = Image.fromarray(arr).convert("RGB") if arr.ndim == 2 else Image.fromarray(arr)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name, "PNG")
        logger.info(f"DICOM converted to PNG: {tmp.name}")
        return tmp.name

    except ImportError:
        raise RuntimeError("pydicom not installed. Run: pip install pydicom numpy pillow")
    except Exception as e:
        raise RuntimeError(f"DICOM conversion failed: {e}")


def _nifti_to_png(nii_path: str) -> str:
    try:
        import nibabel as nib
        import numpy as np
        from PIL import Image

        img  = nib.load(nii_path)
        data = img.get_fdata()

        if data.ndim == 3:
            arr = data[:, :, data.shape[2] // 2]
        elif data.ndim == 4:
            arr = data[:, :, data.shape[2] // 2, data.shape[3] // 2]
        else:
            arr = data

        arr  = arr.astype(float)
        lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
        if hi > lo:
            arr = np.clip((arr - lo) / (hi - lo) * 255, 0, 255)
        arr = arr.astype(np.uint8)

        img_pil = Image.fromarray(arr).convert("RGB")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img_pil.save(tmp.name, "PNG")
        logger.info(f"NIfTI converted to PNG: {tmp.name}")
        return tmp.name

    except ImportError:
        raise RuntimeError("nibabel not installed. Run: pip install nibabel numpy pillow")
    except Exception as e:
        raise RuntimeError(f"NIfTI conversion failed: {e}")


NORMAL_LABELS = {"No Abnormality Detected", "Normal", "Healthy", "No Fracture Detected", "Healthy Skin"}
HIGH_PRIORITY_LABELS = {
    "Glioblastoma (GBM)", "Malignant Glioma", "Astrocytoma", "Brain Metastasis",
    "Lung Adenocarcinoma", "Small Cell Lung Cancer", "Squamous Cell Carcinoma",
    "Hepatocellular Carcinoma (HCC)", "Renal Cell Carcinoma", "Pancreatic Adenocarcinoma",
    "Lymphoma", "Ischemic Stroke", "Hemorrhagic Stroke", "Spinal Tumor / Metastasis",
    "Malignant Mass Detected", "Tuberculosis (TB)", "Pulmonary Fibrosis",
    "Severe Pneumonia", "Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma (Skin)",
}

MODALITY_PROMPTS = {
    "brain_mri": {
        "context": "Brain MRI scan",
        "focus": "brain tumors (glioblastoma with ring enhancement and necrosis, astrocytoma, meningioma, pituitary adenoma, brain metastasis), stroke (ischemic=hypodense, hemorrhagic=hyperdense), MS plaques, hydrocephalus, abscess",
        "classes": ["No Abnormality Detected", "Glioblastoma (GBM)", "Astrocytoma (Grade II-III)", "Meningioma", "Pituitary Adenoma", "Brain Metastasis", "Ischemic Stroke", "Hemorrhagic Stroke", "Multiple Sclerosis (MS)", "Brain Abscess", "Hydrocephalus", "Other Abnormality"],
    },
    "chest_xray": {
        "context": "Chest X-Ray",
        "focus": "lung nodules/masses (spiculated=malignant), consolidation (pneumonia), pleural effusion, mediastinal widening, cardiomegaly, pneumothorax, COVID-19 ground-glass, tuberculosis (upper lobe cavitation), pulmonary fibrosis, atelectasis, pulmonary edema",
        "classes": ["No Abnormality Detected", "Tuberculosis (TB)", "Pneumonia (Bacterial)", "Pneumonia (Viral)", "COVID-19", "Lung Adenocarcinoma", "Squamous Cell Carcinoma", "Small Cell Lung Cancer", "Pulmonary Metastasis", "Pleural Effusion", "Pneumothorax", "Cardiomegaly", "Pulmonary Edema", "Pulmonary Fibrosis", "Atelectasis", "Lymphoma", "Other Abnormality"],
    },
    "ct_scan": {
        "context": "CT scan",
        "focus": "enhancing masses with necrosis, lymphadenopathy, bone destruction, HCC arterial enhancement, renal mass, pancreatic hypodense mass, hemorrhage (hyperdense), stroke (hypodense), appendicitis, aortic aneurysm",
        "classes": ["No Abnormality Detected", "Glioblastoma / Brain Tumor", "Lung Cancer", "Hepatocellular Carcinoma (HCC)", "Renal Cell Carcinoma", "Pancreatic Adenocarcinoma", "Colorectal Cancer", "Lymphoma", "Spinal / Bone Metastasis", "Ischemic Stroke", "Hemorrhagic Stroke", "Aortic Aneurysm", "Appendicitis", "Diverticulitis", "Kidney Stone (Nephrolithiasis)", "Other Abnormality"],
    },
    "mri_spine": {
        "context": "Spine MRI scan",
        "focus": "vertebral metastasis, spinal cord tumor, disc herniation (L4-L5, L5-S1), spinal stenosis, infection/discitis, compression fracture, spondylolisthesis",
        "classes": ["No Abnormality Detected", "Disc Herniation", "Spinal Stenosis", "Vertebral Metastasis", "Spinal Cord Tumor", "Multiple Myeloma", "Compression Fracture", "Spondylolisthesis", "Infection / Discitis", "Syringomyelia", "Other Abnormality"],
    },
    "abdominal": {
        "context": "Abdominal scan",
        "focus": "liver masses, renal mass, pancreatic mass, adrenal mass, lymphadenopathy, ascites, gallstones, appendicitis",
        "classes": ["No Abnormality Detected", "Hepatocellular Carcinoma (HCC)", "Liver Metastasis", "Renal Cell Carcinoma", "Pancreatic Adenocarcinoma", "Adrenal Carcinoma", "Colorectal Cancer", "Peritoneal Carcinomatosis", "Gallstones (Cholelithiasis)", "Liver Cirrhosis", "Splenomegaly", "Benign Finding", "Other Abnormality"],
    },
    "skin": {
        "context": "skin/dermatology photo",
        "focus": "acne, melanoma, basal cell carcinoma, squamous cell carcinoma, psoriasis, eczema, rosacea, vitiligo, ringworm, contact dermatitis, urticaria, herpes, warts, cellulitis, impetigo",
        "classes": ["Healthy Skin", "Acne Vulgaris (Mild)", "Acne Vulgaris (Moderate)", "Acne Vulgaris (Severe/Cystic)", "Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma (Skin)", "Psoriasis", "Eczema / Atopic Dermatitis", "Rosacea", "Vitiligo", "Ringworm / Tinea Corporis", "Contact Dermatitis", "Urticaria (Hives)", "Herpes Simplex / Zoster", "Warts (Verruca)", "Seborrheic Keratosis", "Cellulitis", "Impetigo", "Lipoma / Sebaceous Cyst", "Other Skin Condition"],
    },
    "xray_bone": {
        "context": "X-ray of a bone/joint",
        "focus": "fractures (complete, comminuted, stress, hairline, avulsion, growth plate), dislocations, bone misalignment, soft tissue swelling, joint space narrowing, bone tumors, osteomyelitis",
        "classes": ["No Fracture Detected", "Simple Fracture", "Comminuted Fracture", "Stress Fracture / Hairline Fracture", "Avulsion Fracture", "Dislocation", "Fracture-Dislocation", "Growth Plate Fracture (Salter-Harris)", "Bone Tumor / Lesion", "Osteomyelitis (Bone Infection)", "Osteoporosis / Osteopenia", "Joint Space Narrowing (Arthritis)", "Soft Tissue Injury Only", "Other Musculoskeletal Finding"],
    },
    "general": {
        "context": "medical image",
        "focus": "any visible abnormality: masses, lesions, nodules, tumors, infections, inflammation, fractures, skin conditions, or any pathological finding",
        "classes": ["No Abnormality Detected", "Malignant Mass / Cancer", "Suspicious Lesion", "Infection / Abscess", "Inflammatory Condition", "Fracture / Trauma", "Skin Condition", "Neurological Finding", "Cardiovascular Finding", "Respiratory Finding", "Gastrointestinal Finding", "Musculoskeletal Finding", "Other Abnormality"],
    },
}

TREATMENT_MAP = {
    "Glioblastoma (GBM)": "Surgery (maximal resection) + Temozolomide chemotherapy + Radiation therapy (60 Gy). Bevacizumab for recurrence. Median survival 14-16 months.",
    "Astrocytoma (Grade II-III)": "Surgery + Radiation + Temozolomide. Grade II may be monitored; Grade III requires aggressive treatment. IDH mutation testing guides prognosis.",
    "Meningioma": "Observation for small asymptomatic cases. Surgery for symptomatic/large tumors. Stereotactic radiosurgery (Gamma Knife) for residual/recurrent disease.",
    "Pituitary Adenoma": "Dopamine agonists (cabergoline/bromocriptine) for prolactinomas. Transsphenoidal surgery for non-functioning or resistant adenomas.",
    "Brain Metastasis": "Whole-brain radiation (WBRT) or stereotactic radiosurgery (SRS). Targeted therapy if primary is EGFR/ALK+ lung cancer. Immunotherapy (pembrolizumab).",
    "Ischemic Stroke": "EMERGENCY: IV tPA (alteplase) within 4.5 hours. Mechanical thrombectomy within 24 hours for large vessel occlusion. Antiplatelet + statin long-term.",
    "Hemorrhagic Stroke": "EMERGENCY: Blood pressure control (<140 mmHg). Reverse anticoagulants immediately. Surgical evacuation for large hematomas.",
    "Multiple Sclerosis (MS)": "Disease-modifying therapies: interferons, glatiramer, natalizumab, ocrelizumab. Steroids for acute relapses.",
    "Brain Abscess": "Antibiotics (ceftriaxone + metronidazole) 6-8 weeks. Surgical drainage for large (>2.5cm) abscesses.",
    "Hydrocephalus": "Ventriculoperitoneal (VP) shunt placement. Endoscopic third ventriculostomy (ETV) for obstructive hydrocephalus.",
    "Tuberculosis (TB)": "RIPE regimen: Rifampicin + Isoniazid + Pyrazinamide + Ethambutol for 2 months, then Rifampicin + Isoniazid for 4 months. DOT recommended.",
    "Pneumonia (Bacterial)": "Amoxicillin-clavulanate or Azithromycin for community-acquired. IV antibiotics (ceftriaxone + azithromycin) for severe cases.",
    "Pneumonia (Viral)": "Supportive care: rest, hydration, antipyretics. Oseltamivir for influenza within 48 hours. Oxygen if SpO2 <94%.",
    "COVID-19": "Mild: Isolation, rest, hydration. Moderate-severe: Remdesivir, dexamethasone. Hospitalization for critical cases.",
    "Lung Adenocarcinoma": "Surgery (lobectomy) for early stage. EGFR/ALK targeted therapy. Immunotherapy (pembrolizumab) if PD-L1 >50%.",
    "Small Cell Lung Cancer": "Cisplatin + Etoposide + Radiation for limited stage. Atezolizumab + chemotherapy for extensive stage.",
    "Pulmonary Fibrosis": "Pirfenidone or Nintedanib to slow progression. Pulmonary rehabilitation. Lung transplant for advanced disease.",
    "Pleural Effusion": "Thoracentesis for diagnosis and relief. Treat underlying cause.",
    "Pneumothorax": "Small: Observation. Large/tension: EMERGENCY needle decompression then chest tube.",
    "Cardiomegaly": "ACE inhibitors + beta-blockers for heart failure, diuretics for fluid overload.",
    "Pulmonary Edema": "EMERGENCY: Furosemide IV, oxygen/NIV, nitrates.",
    "Acne Vulgaris (Mild)": "Topical retinoids (tretinoin 0.025%) + benzoyl peroxide 2.5-5%. Daily gentle cleanser, SPF 30+.",
    "Acne Vulgaris (Moderate)": "Topical retinoid + benzoyl peroxide. Oral doxycycline 100mg for 3-6 months.",
    "Acne Vulgaris (Severe/Cystic)": "Oral isotretinoin (Accutane) 0.5-1mg/kg/day for 16-20 weeks.",
    "Melanoma": "URGENT surgical excision with wide margins. Sentinel lymph node biopsy. Anti-PD1 (pembrolizumab/nivolumab) + BRAF inhibitors for Stage III-IV.",
    "Basal Cell Carcinoma": "Surgical excision or Mohs surgery. Topical imiquimod for superficial type.",
    "Squamous Cell Carcinoma (Skin)": "Surgical excision with clear margins. Mohs surgery for high-risk locations.",
    "Psoriasis": "Topical corticosteroids + vitamin D analogues. Phototherapy (UVB). Biologics (adalimumab) for severe.",
    "Eczema / Atopic Dermatitis": "Moisturize frequently. Topical corticosteroids for flares. Dupilumab for severe cases.",
    "Rosacea": "Topical metronidazole or azelaic acid. Oral doxycycline 40mg for moderate.",
    "Vitiligo": "Topical corticosteroids or tacrolimus. Narrowband UVB phototherapy. Ruxolitinib cream.",
    "Ringworm / Tinea Corporis": "Topical clotrimazole or terbinafine for 2-4 weeks.",
    "Contact Dermatitis": "Identify and avoid allergen. Topical corticosteroids. Oral antihistamines for itch.",
    "Urticaria (Hives)": "Non-sedating antihistamines (cetirizine, loratadine) daily. Omalizumab for chronic urticaria.",
    "Herpes Simplex / Zoster": "Antiviral: acyclovir, valacyclovir, or famciclovir within 72 hours of rash onset.",
    "Warts (Verruca)": "Salicylic acid 17-40% daily. Cryotherapy every 2-3 weeks.",
    "Cellulitis": "Oral cephalexin or dicloxacillin. IV vancomycin for severe/MRSA.",
    "Impetigo": "Topical mupirocin ointment. Oral cephalexin for widespread cases.",
    "Seborrheic Keratosis": "Usually benign. Cryotherapy or curettage for cosmetic removal.",
    "Lipoma / Sebaceous Cyst": "Observation for asymptomatic. Surgical excision for large/painful lesions.",
    "Hepatocellular Carcinoma (HCC)": "Surgical resection or liver transplant for early stage. TACE or sorafenib for advanced.",
    "Renal Cell Carcinoma": "Nephrectomy for localized. Sunitinib or cabozantinib for metastatic.",
    "Pancreatic Adenocarcinoma": "Whipple procedure for resectable. Gemcitabine + nab-paclitaxel chemotherapy.",
    "Gallstones (Cholelithiasis)": "Laparoscopic cholecystectomy for symptomatic. ERCP for bile duct stones.",
    "Liver Cirrhosis": "Treat underlying cause. Beta-blockers for portal hypertension. Liver transplant for end-stage.",
    "Kidney Stone (Nephrolithiasis)": "Small: hydration + tamsulosin. Larger: ESWL or ureteroscopy.",
    "Disc Herniation": "NSAIDs, physiotherapy, epidural steroids. Microdiscectomy for neurological deficits.",
    "Spinal Stenosis": "Physiotherapy, NSAIDs, epidural steroids. Laminectomy for severe cases.",
    "Compression Fracture": "Analgesics and bracing. Vertebroplasty or kyphoplasty for painful fractures.",
    "Simple Fracture": "Immobilization with cast or splint 4-8 weeks. NSAIDs for pain.",
    "Comminuted Fracture": "URGENT orthopedic referral. Surgical fixation (ORIF) required.",
    "Stress Fracture / Hairline Fracture": "Rest 6-8 weeks. Protective boot for lower limb. Calcium and Vitamin D.",
    "Avulsion Fracture": "Splint/cast 4-6 weeks. Surgical reattachment for large displaced fragments.",
    "Dislocation": "URGENT reduction within 6 hours. Immobilization 3-6 weeks. Physical therapy.",
    "Fracture-Dislocation": "URGENT surgical emergency. ORIF surgery required.",
    "Growth Plate Fracture (Salter-Harris)": "URGENT pediatric orthopedic referral. Surgical fixation for Type III-V.",
    "Osteomyelitis (Bone Infection)": "IV antibiotics 4-6 weeks. Surgical debridement for chronic infection.",
    "Osteoporosis / Osteopenia": "Bisphosphonates (alendronate). Calcium 1000mg/day + Vitamin D 800 IU/day.",
    "Joint Space Narrowing (Arthritis)": "NSAIDs, physiotherapy. Joint replacement for end-stage disease.",
    "Bone Tumor / Lesion": "URGENT oncology referral. MRI + biopsy. Chemotherapy + surgery for malignant.",
}


def _assign_priority(label: str, confidence: float) -> str:
    if label in NORMAL_LABELS and confidence >= 0.70:
        return "LOW"
    if label in HIGH_PRIORITY_LABELS and confidence >= 0.55:
        return "HIGH"
    if label in NORMAL_LABELS:
        return "LOW"
    return "MEDIUM"


def _encode_image_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_prompt(scan_type: str) -> str:
    m = MODALITY_PROMPTS.get(scan_type, MODALITY_PROMPTS["general"])
    classes_str = "\n".join(f"  - {c}" for c in m["classes"])
    return (
        "You are a board-certified radiologist AI with 20 years of experience.\n"
        f"You are analyzing a {m['context']} image.\n\n"
        "CRITICAL RULES:\n"
        "1. Choose a LABEL from the provided list only.\n"
        "2. CONFIDENCE must reflect objective image evidence (clear findings = 0.80-0.95).\n"
        "3. Be CONSISTENT — same image always produces same diagnosis.\n"
        "4. Do NOT default to 'No Abnormality Detected' unless image is genuinely clear.\n"
        "5. Do NOT report a finding without clear visual evidence.\n\n"
        f"Look for: {m['focus']}\n\n"
        "Assess before concluding:\n"
        "- Density/signal changes (hyperdense, hypodense, enhancement)\n"
        "- Structural abnormalities (mass, nodule, consolidation, effusion)\n"
        "- Symmetry and anatomical variants\n"
        "- Size, borders, surrounding tissue involvement\n"
        "- Overall image quality\n\n"
        "Reply in EXACTLY this 7-line format, no extra text:\n"
        "LABEL: <exact diagnosis from list>\n"
        "CONFIDENCE: <0.0 to 1.0>\n"
        "PRIORITY: <HIGH or MEDIUM or LOW>\n"
        "FINDINGS: <3-4 sentences: what you see, where, why this label>\n"
        "CONDITION: <specific disease name>\n"
        "SEVERITY: <Mild or Moderate or Severe or Critical or Early or Advanced or Unable to determine>\n"
        "TREATMENT: <first-line treatment with drug names, dosages, follow-up>\n\n"
        f"LABEL must be one of:\n{classes_str}\n\n"
        "Priority: HIGH=malignant/emergency. MEDIUM=needs treatment. LOW=normal/minor."
    )


def _extract_result(text: str, model: str) -> dict:
    def get_field(key, default=""):
        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith(key + ":"):
                val = line[len(key)+1:].strip()
                if val:
                    return val
        return default

    if "LABEL:" in text.upper() or "FINDINGS:" in text.upper():
        label_v  = get_field("LABEL", "Other Abnormality")
        conf_raw = get_field("CONFIDENCE", "0.75")
        prio_v   = get_field("PRIORITY", "MEDIUM").upper()
        find_v   = get_field("FINDINGS", "")
        cond_v   = get_field("CONDITION", label_v)
        sev_v    = get_field("SEVERITY", "Unable to determine")
        treat_v  = get_field("TREATMENT", "")

        if prio_v not in ("HIGH", "MEDIUM", "LOW"):
            prio_v = "MEDIUM"

        try:
            conf_v = max(0.0, min(1.0, float(conf_raw)))
        except Exception:
            conf_v = 0.75

        logger.info(f"Parse success [{model}]: {label_v}")
        return {
            "result_label":     label_v,
            "confidence_score": conf_v,
            "priority":         prio_v,
            "raw_scores":       {label_v: conf_v},
            "findings":         find_v,
            "cancer_type":      cond_v,
            "stage_estimate":   sev_v,
            "recommendation":   treat_v,
        }

    try:
        if "```" in text:
            chunks = text.split("```")
            text = chunks[1] if len(chunks) > 1 else text
            if text.lower().startswith("json"):
                text = text[4:]
        text = text.strip()
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1:
            text = text[s:e+1]
        result = json.loads(text)
        logger.info(f"JSON fallback success [{model}]")
        return result
    except Exception as ex:
        logger.warning(f"Both parsers failed [{model}]: {ex}")
        return {
            "result_label":     "Other Abnormality",
            "confidence_score": 0.5,
            "priority":         "MEDIUM",
            "raw_scores":       {"Other Abnormality": 0.5},
            "findings":         text[:300] if text else "Unable to parse AI response.",
            "cancer_type":      "Unknown",
            "stage_estimate":   "Unable to determine",
            "recommendation":   "Please consult a specialist for review.",
        }


def _call_gemini_vision(file_path: str, ext: str, scan_type: str = "general") -> dict:
    api_keys = [
        os.environ.get("GEMINI_API_KEY", ""),
        os.environ.get("GEMINI_API_KEY_2", ""),
        os.environ.get("GEMINI_API_KEY_3", ""),
    ]
    api_keys = [k for k in api_keys if k]
    if not api_keys:
        raise RuntimeError("No GEMINI_API_KEY set")

    prompt    = _build_prompt(scan_type)
    image_b64 = _encode_image_base64(file_path)
    mime_type = "image/jpeg" if ext.lower() in ("jpg", "jpeg") else "image/png"

    payload = json.dumps({
        "contents": [{"parts": [
            {"inline_data": {"mime_type": mime_type, "data": image_b64}},
            {"text": prompt},
        ]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1024},
    }).encode("utf-8")

    last_error = "No keys tried"

    for api_key in api_keys:
        key_preview = api_key[:12] + "..."
        logger.info(f"Trying Gemini key: {key_preview}")
        for model in GEMINI_MODELS:
            url = f"{GEMINI_API_BASE}/{model}:generateContent?key={api_key}"
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                text   = raw["candidates"][0]["content"]["parts"][0]["text"].strip()
                result = _extract_result(text, model)
                label  = result.get("result_label", "")
                if label in TREATMENT_MAP and not result.get("recommendation"):
                    result["recommendation"] = TREATMENT_MAP[label]
                return result
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8")
                last_error = f"{model} HTTP {e.code}: {body[:200]}"
                if e.code in (429, 404):
                    logger.warning(f"Model {model} failed with {e.code} — trying next")
                    continue
                raise RuntimeError(last_error)
            except Exception as e:
                last_error = f"{model}: {e}"
                continue

    raise RuntimeError(f"All Gemini keys and models exhausted. Last: {last_error}")


def _fallback_simulation(ext: str, error_msg: str = "") -> dict:
    api_key_set = bool(
        os.environ.get("GEMINI_API_KEY") or
        os.environ.get("GEMINI_API_KEY_2") or
        os.environ.get("GEMINI_API_KEY_3")
    )
    if not api_key_set:
        findings = "GEMINI_API_KEY is not set in your .env file."
        rec      = "Get a free key at https://aistudio.google.com"
    else:
        findings = f"Gemini API call failed: {error_msg[:200]}"
        rec      = "Please create a fresh API key at https://aistudio.google.com"

    return {
        "result_label":     "SIMULATION — No Real Analysis",
        "confidence_score": 0.0,
        "priority":         "MEDIUM",
        "raw_scores":       {"SIMULATION — No Real Analysis": 1.0},
        "findings":         findings,
        "cancer_type":      "Not analyzed",
        "stage_estimate":   "Not applicable",
        "recommendation":   rec,
    }


def run_inference(
    file_path:  str,
    upload_id:  int,
    ext:        str,
    model_name: str = "gemini-vision",
    scan_type:  str = "general",
) -> Prediction:
    logger.info(f"Gemini Vision inference: {file_path} | ext={ext} | type={scan_type}")

    ext_clean    = ext.lower().lstrip(".")
    tmp_png_path = None

    if ext_clean not in VISION_SUPPORTED:
        result = _fallback_simulation(ext_clean, f"Unsupported file type: {ext_clean}")
    elif not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY_2") or os.environ.get("GEMINI_API_KEY_3")):
        result = _fallback_simulation(ext_clean)
    else:
        try:
            if ext_clean == "dcm":
                logger.info("Converting DICOM to PNG for Gemini Vision...")
                tmp_png_path = _dicom_to_png(file_path)
                vision_path  = tmp_png_path
                vision_ext   = "png"
                preview_png  = file_path.rsplit(".", 1)[0] + "_preview.png"
                shutil.copy2(tmp_png_path, preview_png)
                logger.info(f"DICOM preview saved: {preview_png}")

            elif ext_clean in ("nii", "gz", "nii.gz"):
                logger.info("Converting NIfTI to PNG for Gemini Vision...")
                tmp_png_path = _nifti_to_png(file_path)
                vision_path  = tmp_png_path
                vision_ext   = "png"
                preview_png  = file_path.rsplit(".", 1)[0] + "_preview.png"
                shutil.copy2(tmp_png_path, preview_png)
                logger.info(f"NIfTI preview saved: {preview_png}")

            else:
                vision_path = file_path
                vision_ext  = ext_clean

            result = _call_gemini_vision(vision_path, vision_ext, scan_type)

        except Exception as exc:
            logger.error(f"Gemini Vision failed: {exc}")
            result = _fallback_simulation(ext_clean, str(exc))
        finally:
            if tmp_png_path and os.path.exists(tmp_png_path):
                try:
                    os.remove(tmp_png_path)
                except Exception:
                    pass

    label          = result["result_label"]
    confidence     = float(result.get("confidence_score", 0.0))
    priority       = result.get("priority") or _assign_priority(label, confidence)
    findings       = result.get("findings", "")
    cancer_type    = result.get("cancer_type", label)
    stage_estimate = result.get("stage_estimate", "")
    recommendation = result.get("recommendation", "")

    if label in TREATMENT_MAP and (not recommendation or len(recommendation) < 30):
        recommendation = TREATMENT_MAP[label]

    raw_scores = {
        k: round(float(v), 4)
        for k, v in result.get("raw_scores", {}).items()
        if isinstance(v, (int, float))
    }

    extra             = f"{findings} ||| {cancer_type} ||| {stage_estimate} ||| {recommendation}"
    model_version_str = f"gemini-vision | {extra}"

    logger.info(f"Result: {label} ({confidence:.2%}) | Priority: {priority}")

    return Prediction(
        upload_id        = upload_id,
        result_label     = label,
        confidence_score = round(confidence, 4),
        raw_scores       = json.dumps(raw_scores),
        priority         = priority,
        heatmap_path     = None,
        analysis_date    = datetime.utcnow(),
        model_version    = model_version_str,
    )


def batch_infer(file_paths: list, model_name: str = "gemini-vision", scan_type: str = "general") -> list:
    results = []
    for fp in file_paths:
        ext = fp.rsplit(".", 1)[-1].lower().lstrip(".")
        try:
            if ext in VISION_SUPPORTED:
                result = _call_gemini_vision(fp, ext, scan_type)
            else:
                result = _fallback_simulation(ext)
            results.append((result["result_label"], float(result["confidence_score"]), result["raw_scores"]))
        except Exception as exc:
            logger.error(f"Batch inference failed for {fp}: {exc}")
            results.append(("Error", 0.0, {}))
    return results