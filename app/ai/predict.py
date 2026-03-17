"""
app/ai/predict.py  —  Medical AI Inference Pipeline  (v2.2)

Fix in v2.2 vs v2.1:
  NeuroimagingValidator.validate() now uses a negated() helper so that
  sentences like "no evidence of compressed sulci" do NOT falsely trigger
  the hydrocephalus rule correction.
"""

import os
import re
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

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VISION_SUPPORTED = {"png", "jpg", "jpeg", "dcm", "nii", "gz", "nii.gz"}

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
]

DIFFERENTIAL_THRESHOLD = 0.70

SAFETY_DISCLAIMER = (
    "⚠️ MEDICAL DISCLAIMER: This AI analysis is a clinical decision-support tool ONLY. "
    "It is NOT a substitute for professional medical diagnosis or radiological interpretation. "
    "All findings must be reviewed and confirmed by a licensed radiologist or physician "
    "before any clinical action is taken."
)

# ─────────────────────────────────────────────────────────────────────────────
# Priority label sets
# ─────────────────────────────────────────────────────────────────────────────

NORMAL_LABELS = {
    "No Abnormality Detected", "Normal", "Healthy",
    "No Fracture Detected", "Healthy Skin",
}

HIGH_PRIORITY_LABELS = {
    "Glioblastoma (GBM)", "Malignant Glioma", "Astrocytoma", "Brain Metastasis",
    "Lung Adenocarcinoma", "Small Cell Lung Cancer", "Squamous Cell Carcinoma",
    "Hepatocellular Carcinoma (HCC)", "Renal Cell Carcinoma",
    "Pancreatic Adenocarcinoma", "Lymphoma",
    "Ischemic Stroke", "Hemorrhagic Stroke",
    "Hydrocephalus (Obstructive)",
    "Spinal Tumor / Metastasis", "Malignant Mass Detected",
    "Tuberculosis (TB)", "Pulmonary Fibrosis", "Severe Pneumonia",
    "Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma (Skin)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Modality prompts
# ─────────────────────────────────────────────────────────────────────────────

MODALITY_PROMPTS = {
    "brain_mri": {
        "context": "Brain MRI scan",
        "focus": (
            "ALZHEIMER'S DISEASE vs HYDROCEPHALUS — read this carefully before deciding:\n\n"
            "Alzheimer's Disease features (ALL of these point to AD):\n"
            "  • Cortical atrophy concentrated in temporal and parietal lobes\n"
            "  • Hippocampal atrophy — this is the STRONGEST single discriminator\n"
            "  • WIDENED sulci (sulci are open, enlarged, prominent — due to brain shrinkage)\n"
            "  • Ventricular enlargement that is PROPORTIONAL to the cortical loss (ex vacuo)\n"
            "  • No periventricular T2/FLAIR signal change\n\n"
            "Hydrocephalus features (ALL of these point to HCP):\n"
            "  • DISPROPORTIONATE ventricular enlargement (much larger than cortical atrophy warrants)\n"
            "  • Sulci are COMPRESSED, EFFACED, pushed together — NOT widened\n"
            "  • Periventricular T2/FLAIR hyperintensity (CSF seeping through ependyma)\n"
            "  • Rounded, ballooned frontal horns under pressure\n"
            "  • Evans index > 0.3\n\n"
            "DECISION RULE — apply this before choosing a label:\n"
            "  IF sulci are WIDENED → Alzheimer's Disease (not Hydrocephalus)\n"
            "  IF sulci are COMPRESSED → Hydrocephalus (not Alzheimer's)\n"
            "  IF periventricular edema present → strongly favors Hydrocephalus\n"
            "  IF hippocampal atrophy present → strongly favors Alzheimer's\n\n"
            "Also assess: brain tumors (glioblastoma with ring enhancement + necrosis, "
            "astrocytoma, meningioma, pituitary adenoma, brain metastasis), "
            "stroke (ischemic=hypodense, hemorrhagic=hyperdense), MS plaques, abscess"
        ),
        "classes": [
            "No Abnormality Detected",
            "Alzheimer's Disease (Mild)",
            "Alzheimer's Disease (Moderate)",
            "Alzheimer's Disease (Severe)",
            "Hydrocephalus (Normal Pressure / NPH)",
            "Hydrocephalus (Obstructive)",
            "Glioblastoma (GBM)",
            "Astrocytoma (Grade II-III)",
            "Meningioma",
            "Pituitary Adenoma",
            "Brain Metastasis",
            "Ischemic Stroke",
            "Hemorrhagic Stroke",
            "Multiple Sclerosis (MS)",
            "Brain Abscess",
            "Other Abnormality",
        ],
    },

    "chest_xray": {
        "context": "Chest X-Ray",
        "focus": (
            "lung nodules/masses (spiculated=malignant), consolidation (pneumonia), "
            "pleural effusion, mediastinal widening, cardiomegaly, pneumothorax, "
            "COVID-19 ground-glass opacities, tuberculosis (upper lobe cavitation), "
            "pulmonary fibrosis (reticular pattern), atelectasis, pulmonary edema"
        ),
        "classes": [
            "No Abnormality Detected", "Tuberculosis (TB)",
            "Pneumonia (Bacterial)", "Pneumonia (Viral)", "COVID-19",
            "Lung Adenocarcinoma", "Squamous Cell Carcinoma", "Small Cell Lung Cancer",
            "Pulmonary Metastasis", "Pleural Effusion", "Pneumothorax",
            "Cardiomegaly", "Pulmonary Edema", "Pulmonary Fibrosis",
            "Atelectasis", "Lymphoma", "Other Abnormality",
        ],
    },

    "ct_scan": {
        "context": "CT scan",
        "focus": (
            "enhancing masses with necrosis, lymphadenopathy, bone destruction, "
            "HCC arterial enhancement, renal mass, pancreatic hypodense mass, "
            "hemorrhage (hyperdense), stroke (hypodense), appendicitis, aortic aneurysm"
        ),
        "classes": [
            "No Abnormality Detected", "Glioblastoma / Brain Tumor",
            "Lung Cancer", "Hepatocellular Carcinoma (HCC)", "Renal Cell Carcinoma",
            "Pancreatic Adenocarcinoma", "Colorectal Cancer", "Lymphoma",
            "Spinal / Bone Metastasis", "Ischemic Stroke", "Hemorrhagic Stroke",
            "Aortic Aneurysm", "Appendicitis", "Diverticulitis",
            "Kidney Stone (Nephrolithiasis)", "Other Abnormality",
        ],
    },

    "mri_spine": {
        "context": "Spine MRI scan",
        "focus": (
            "vertebral metastasis, spinal cord tumor, disc herniation (L4-L5, L5-S1), "
            "spinal stenosis, infection/discitis, compression fracture, spondylolisthesis"
        ),
        "classes": [
            "No Abnormality Detected", "Disc Herniation", "Spinal Stenosis",
            "Vertebral Metastasis", "Spinal Cord Tumor", "Multiple Myeloma",
            "Compression Fracture", "Spondylolisthesis", "Infection / Discitis",
            "Syringomyelia", "Other Abnormality",
        ],
    },

    "abdominal": {
        "context": "Abdominal scan",
        "focus": (
            "liver masses, renal mass, pancreatic mass, adrenal mass, "
            "lymphadenopathy, ascites, gallstones, appendicitis"
        ),
        "classes": [
            "No Abnormality Detected", "Hepatocellular Carcinoma (HCC)",
            "Liver Metastasis", "Renal Cell Carcinoma", "Pancreatic Adenocarcinoma",
            "Adrenal Carcinoma", "Colorectal Cancer", "Peritoneal Carcinomatosis",
            "Gallstones (Cholelithiasis)", "Liver Cirrhosis", "Splenomegaly",
            "Benign Finding", "Other Abnormality",
        ],
    },

    "skin": {
        "context": "skin/dermatology photo",
        "focus": (
            "acne, melanoma, basal cell carcinoma, squamous cell carcinoma, "
            "psoriasis, eczema, rosacea, vitiligo, ringworm, contact dermatitis, "
            "urticaria, herpes, warts, cellulitis, impetigo"
        ),
        "classes": [
            "Healthy Skin", "Acne Vulgaris (Mild)", "Acne Vulgaris (Moderate)",
            "Acne Vulgaris (Severe/Cystic)", "Melanoma", "Basal Cell Carcinoma",
            "Squamous Cell Carcinoma (Skin)", "Psoriasis", "Eczema / Atopic Dermatitis",
            "Rosacea", "Vitiligo", "Ringworm / Tinea Corporis", "Contact Dermatitis",
            "Urticaria (Hives)", "Herpes Simplex / Zoster", "Warts (Verruca)",
            "Seborrheic Keratosis", "Cellulitis", "Impetigo",
            "Lipoma / Sebaceous Cyst", "Other Skin Condition",
        ],
    },

    "xray_bone": {
        "context": "X-ray of a bone/joint",
        "focus": (
            "fractures (complete, comminuted, stress, hairline, avulsion, growth plate), "
            "dislocations, bone misalignment, soft tissue swelling, joint space narrowing, "
            "bone tumors, osteomyelitis"
        ),
        "classes": [
            "No Fracture Detected", "Simple Fracture", "Comminuted Fracture",
            "Stress Fracture / Hairline Fracture", "Avulsion Fracture",
            "Dislocation", "Fracture-Dislocation",
            "Growth Plate Fracture (Salter-Harris)", "Bone Tumor / Lesion",
            "Osteomyelitis (Bone Infection)", "Osteoporosis / Osteopenia",
            "Joint Space Narrowing (Arthritis)", "Soft Tissue Injury Only",
            "Other Musculoskeletal Finding",
        ],
    },

    "general": {
        "context": "medical image",
        "focus": (
            "any visible abnormality: masses, lesions, nodules, tumors, infections, "
            "inflammation, fractures, skin conditions, or any pathological finding"
        ),
        "classes": [
            "No Abnormality Detected", "Malignant Mass / Cancer",
            "Suspicious Lesion", "Infection / Abscess", "Inflammatory Condition",
            "Fracture / Trauma", "Skin Condition", "Neurological Finding",
            "Cardiovascular Finding", "Respiratory Finding",
            "Gastrointestinal Finding", "Musculoskeletal Finding", "Other Abnormality",
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Treatment map
# ─────────────────────────────────────────────────────────────────────────────

TREATMENT_MAP = {
    "Alzheimer's Disease (Mild)": (
        "Donepezil 5–10 mg/day (first-line AChEI). Cognitive rehabilitation, "
        "physical activity program. MMSE/MoCA baseline. Reassess in 6 months. "
        "Caregiver education and support."
    ),
    "Alzheimer's Disease (Moderate)": (
        "Donepezil 10 mg/day OR Rivastigmine patch 9.5 mg/24h + Memantine 10–20 mg/day. "
        "Manage BPSD: low-dose antipsychotics only for severe agitation. "
        "Advance care planning. Occupational therapy."
    ),
    "Alzheimer's Disease (Severe)": (
        "Memantine 20 mg/day + cholinesterase inhibitor. Palliative approach: "
        "comfort-focused care, swallowing assessment. Multidisciplinary team: "
        "geriatrics + palliative care."
    ),
    "Hydrocephalus (Normal Pressure / NPH)": (
        "Large-volume LP (30–50 ml CSF) to confirm tap-test responsiveness. "
        "VP shunt with programmable pressure valve. Monitor Evans index on serial imaging. "
        "Target clinical triad: gait + continence + cognition."
    ),
    "Hydrocephalus (Obstructive)": (
        "URGENT neurosurgical referral. Endoscopic Third Ventriculostomy (ETV) "
        "for aqueductal stenosis. VP shunt for non-ETV candidates. "
        "Identify and treat underlying cause (tumor, cyst, hemorrhage)."
    ),
    "Glioblastoma (GBM)": (
        "Maximal safe resection + Temozolomide 75 mg/m² + Radiation 60 Gy/30 fr. "
        "Adjuvant Temozolomide 150–200 mg/m² ×5 days/28-day cycle. "
        "Bevacizumab for recurrence. MGMT/IDH/EGFR testing mandatory."
    ),
    "Astrocytoma (Grade II-III)": (
        "Surgery + Radiation + Temozolomide. Grade II: watch-and-wait post-resection if low-risk. "
        "Grade III: aggressive treatment. IDH mutation guides prognosis."
    ),
    "Meningioma": (
        "Observation for small asymptomatic tumors. Surgery for symptomatic/large lesions. "
        "Stereotactic radiosurgery (Gamma Knife) for residual or recurrent disease."
    ),
    "Pituitary Adenoma": (
        "Prolactinomas: cabergoline 0.5 mg twice weekly (first-line). "
        "Non-functioning or resistant: transsphenoidal surgery."
    ),
    "Brain Metastasis": (
        "Stereotactic radiosurgery (SRS) for ≤3 lesions. WBRT for multiple lesions. "
        "Targeted therapy if primary is EGFR/ALK+ lung cancer. "
        "Pembrolizumab if PD-L1 positive."
    ),
    "Ischemic Stroke": (
        "EMERGENCY: IV alteplase (tPA) within 4.5 hours of onset. "
        "Mechanical thrombectomy within 24 hours for large vessel occlusion. "
        "Dual antiplatelet (aspirin + clopidogrel) + high-intensity statin long-term. "
        "BP target <180/105 in acute phase."
    ),
    "Hemorrhagic Stroke": (
        "EMERGENCY: Systolic BP target <140 mmHg. Reverse anticoagulants immediately "
        "(PCC for warfarin; idarucizumab for dabigatran). "
        "Neurosurgery ICU. Surgical evacuation for cerebellar hematoma >3 cm or herniation."
    ),
    "Multiple Sclerosis (MS)": (
        "Disease-modifying therapy: interferons, glatiramer, natalizumab, or ocrelizumab. "
        "IV methylprednisolone 1 g/day ×3–5 days for acute relapses."
    ),
    "Brain Abscess": (
        "IV ceftriaxone 2 g q12h + metronidazole 500 mg q8h for 6–8 weeks. "
        "Surgical drainage for abscesses >2.5 cm or mass effect."
    ),
    "Tuberculosis (TB)": (
        "RIPE: Rifampicin + Isoniazid + Pyrazinamide + Ethambutol for 2 months, "
        "then Rifampicin + Isoniazid for 4 months. DOT recommended. Notify public health."
    ),
    "Pneumonia (Bacterial)": (
        "Amoxicillin-clavulanate or Azithromycin for mild community-acquired. "
        "IV ceftriaxone + azithromycin for moderate-severe. De-escalate per culture."
    ),
    "Pneumonia (Viral)": (
        "Supportive: rest, hydration, antipyretics. Oseltamivir for influenza within 48 h. "
        "Oxygen if SpO2 <94%."
    ),
    "COVID-19": (
        "Mild: isolation, rest, hydration, paracetamol. "
        "Moderate-severe: Remdesivir + dexamethasone 6 mg/day. "
        "Hospitalization for critical cases with oxygen/NIV."
    ),
    "Lung Adenocarcinoma": (
        "Surgery (lobectomy) for early stage. EGFR/ALK/ROS1 testing mandatory. "
        "Targeted therapy (osimertinib for EGFR+). "
        "Pembrolizumab if PD-L1 ≥50%. Platinum doublet for others."
    ),
    "Small Cell Lung Cancer": (
        "Limited stage: cisplatin + etoposide + concurrent thoracic RT. "
        "Extensive stage: atezolizumab + carboplatin + etoposide. "
        "Prophylactic cranial irradiation for responders."
    ),
    "Pulmonary Fibrosis": (
        "Pirfenidone 2403 mg/day OR nintedanib 150 mg twice daily to slow progression. "
        "Pulmonary rehabilitation. Lung transplant evaluation for advanced disease."
    ),
    "Pleural Effusion": (
        "Diagnostic thoracentesis. Treat underlying cause. "
        "Pleurodesis for recurrent malignant effusion."
    ),
    "Pneumothorax": (
        "Small (<2 cm): observation + oxygen. "
        "Large/tension: EMERGENCY needle decompression (2nd ICS MCL) then chest tube."
    ),
    "Cardiomegaly": (
        "ACE inhibitors + beta-blockers for heart failure. Loop diuretics (furosemide) "
        "for fluid overload. Echo and cardiology referral."
    ),
    "Pulmonary Edema": (
        "EMERGENCY: IV furosemide 40–80 mg. High-flow oxygen or NIV (CPAP). "
        "IV nitrates for BP management. Treat underlying cause."
    ),
    "Melanoma": (
        "URGENT wide local excision with 1–2 cm margins. "
        "Sentinel lymph node biopsy if ≥T1b. "
        "Anti-PD1 (pembrolizumab/nivolumab) ± BRAF inhibitors for Stage III–IV."
    ),
    "Basal Cell Carcinoma": (
        "Mohs surgery for high-risk/facial lesions. "
        "Standard excision for low-risk lesions. "
        "Topical imiquimod for superficial type."
    ),
    "Squamous Cell Carcinoma (Skin)": (
        "Surgical excision with clear 4–6 mm margins. "
        "Mohs surgery for high-risk locations. "
        "Cemiplimab for locally advanced or metastatic disease."
    ),
    "Acne Vulgaris (Mild)": (
        "Topical tretinoin 0.025% + benzoyl peroxide 2.5–5%. SPF 30+ daily."
    ),
    "Acne Vulgaris (Moderate)": (
        "Topical retinoid + benzoyl peroxide. Oral doxycycline 100 mg/day for 3–6 months."
    ),
    "Acne Vulgaris (Severe/Cystic)": (
        "Oral isotretinoin 0.5–1 mg/kg/day for 16–20 weeks. "
        "Pregnancy prevention mandatory. Monthly LFT and lipid monitoring."
    ),
    "Psoriasis": (
        "Mild: topical corticosteroids + calcipotriol. "
        "Moderate: narrowband UVB phototherapy. "
        "Severe: biologics (adalimumab, secukinumab, ixekizumab)."
    ),
    "Eczema / Atopic Dermatitis": (
        "Frequent emollient use. Topical corticosteroids for flares. "
        "Tacrolimus ointment for face/sensitive areas. "
        "Dupilumab 300 mg q2w for moderate-severe."
    ),
    "Hepatocellular Carcinoma (HCC)": (
        "Surgical resection or liver transplant (Milan criteria) for early stage. "
        "TACE for intermediate stage. Sorafenib 400 mg twice daily for advanced."
    ),
    "Renal Cell Carcinoma": (
        "Partial/radical nephrectomy for localized disease. "
        "Sunitinib or cabozantinib for metastatic. "
        "Nivolumab + ipilimumab for intermediate/poor risk."
    ),
    "Pancreatic Adenocarcinoma": (
        "Whipple procedure (pancreaticoduodenectomy) for resectable disease. "
        "Gemcitabine + nab-paclitaxel or FOLFIRINOX for metastatic."
    ),
    "Gallstones (Cholelithiasis)": (
        "Laparoscopic cholecystectomy for symptomatic gallstones. "
        "ERCP + sphincterotomy for common bile duct stones."
    ),
    "Liver Cirrhosis": (
        "Treat underlying cause (antivirals for HBV/HCV, abstinence for alcohol). "
        "Beta-blockers (propranolol) for portal hypertension. "
        "Liver transplant evaluation for Child-Pugh C."
    ),
    "Kidney Stone (Nephrolithiasis)": (
        "Small (<5 mm): hydration + tamsulosin 0.4 mg/day for medical expulsion. "
        "5–10 mm: ESWL. >10 mm or obstructing: ureteroscopy."
    ),
    "Disc Herniation": (
        "NSAIDs + physiotherapy for 6 weeks. Epidural steroid injection if not improving. "
        "Microdiscectomy for neurological deficits or cauda equina."
    ),
    "Spinal Stenosis": (
        "Physiotherapy, NSAIDs, epidural steroids. "
        "Laminectomy/decompression for severe neurogenic claudication."
    ),
    "Compression Fracture": (
        "Analgesia + thoracolumbar brace. "
        "Vertebroplasty or kyphoplasty for painful refractory fractures."
    ),
    "Simple Fracture": (
        "Immobilisation with cast or splint 4–8 weeks. NSAIDs for pain. "
        "Orthopaedic follow-up at 1 week."
    ),
    "Comminuted Fracture": (
        "URGENT orthopaedic referral. Surgical fixation (ORIF) typically required."
    ),
    "Stress Fracture / Hairline Fracture": (
        "Rest 6–8 weeks. Protective boot for lower limb. "
        "Calcium 1000 mg/day + Vitamin D 800 IU/day. Activity modification."
    ),
    "Avulsion Fracture": (
        "Splint/cast 4–6 weeks for non-displaced. "
        "Surgical reattachment for large displaced fragments."
    ),
    "Dislocation": (
        "URGENT closed reduction within 6 hours under sedation/analgesia. "
        "Post-reduction immobilisation 3–6 weeks. Physical therapy."
    ),
    "Fracture-Dislocation": (
        "URGENT surgical emergency. ORIF surgery required. "
        "Vascular and nerve status must be assessed immediately."
    ),
    "Growth Plate Fracture (Salter-Harris)": (
        "URGENT paediatric orthopaedic referral. "
        "Type I-II: cast immobilisation. Type III-V: surgical fixation."
    ),
    "Osteomyelitis (Bone Infection)": (
        "IV antibiotics 4–6 weeks (cefazolin for MSSA; vancomycin for MRSA). "
        "Surgical debridement for chronic infection or abscess."
    ),
    "Osteoporosis / Osteopenia": (
        "Bisphosphonates: alendronate 70 mg/week. "
        "Calcium 1000–1200 mg/day + Vitamin D 800–1000 IU/day. "
        "DEXA scan baseline and repeat in 2 years."
    ),
    "Joint Space Narrowing (Arthritis)": (
        "NSAIDs + physiotherapy. Intra-articular corticosteroids. "
        "Total joint replacement for end-stage disease."
    ),
    "Bone Tumor / Lesion": (
        "URGENT oncology referral. MRI + biopsy for characterisation. "
        "Chemotherapy (doxorubicin + cisplatin) + surgery for osteosarcoma."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Rule-based neuroimaging validator
# ─────────────────────────────────────────────────────────────────────────────

class NeuroimagingValidator:
    """
    Post-processing rule engine.

    v2.2 fix: uses negated() so that "no evidence of compressed sulci"
    does NOT falsely trigger the Hydrocephalus correction rule.
    """

    ALZHEIMER_KW = [
        "cortical atrophy", "hippocampal atrophy", "hippocampal volume",
        "widened sulci", "sulcal enlargement", "parietal atrophy",
        "temporal atrophy", "entorhinal", "ex vacuo", "parenchymal loss",
        "generalized atrophy", "prominent sulci",
    ]
    HYDRO_KW = [
        "disproportionate ventricular", "periventricular edema",
        "periventricular t2", "periventricular flair",
        "rounded ventricles", "evans index", "ballooned",
        "csf seepage", "nph", "corpus callosum bowing",
        "obstructive hydrocephalus",
    ]
    COMPRESSION_KW = [
        "compressed sulci", "sulci compressed", "sulci effaced",
        "effacement of sulci", "sulci pushed", "sulci obliterated",
        "sulcal effacement",
    ]
    WIDENING_KW = [
        "widened sulci", "sulci widened", "enlarged sulci",
        "prominent sulci", "sulcal enlargement", "open sulci",
    ]

    # Words that negate a nearby keyword — look up to 8 words before the keyword
    _NEGATION_PATTERN = re.compile(
        r'\b(no|not|without|absence of|no evidence of|no signs? of|no findings? of'
        r'|does not|do not|did not|cannot|ruled out)\b'
        r'[\w\s,]{0,50}?'
    )

    @classmethod
    def _is_negated(cls, keyword: str, text: str) -> bool:
        """
        Returns True if the keyword appears in the text but is preceded by
        a negation word within 50 characters, meaning its meaning is reversed.

        Example:
            "no evidence of compressed sulci"  → negated  → True
            "compressed sulci are present"     → not negated → False
        """
        idx = text.find(keyword)
        if idx == -1:
            return False
        # Look at the 60 characters immediately before the keyword
        window = text[max(0, idx - 60): idx]
        return bool(cls._NEGATION_PATTERN.search(window))

    @classmethod
    def _keyword_active(cls, keyword: str, text: str) -> bool:
        """True if keyword is present AND not negated."""
        return keyword in text and not cls._is_negated(keyword, text)

    @classmethod
    def validate(cls, result: dict, scan_type: str) -> dict:
        """
        Applies rules to result dict.
        Returns the same dict with label/confidence possibly corrected
        and validation_notes populated.
        """
        if scan_type not in ("brain_mri", "ct_scan", "general"):
            result.setdefault("validation_notes", "")
            result["disclaimer"] = SAFETY_DISCLAIMER
            return result

        label      = result.get("result_label", "")
        confidence = float(result.get("confidence_score", 0.75))
        findings   = result.get("findings", "").lower()
        notes      = []

        # ── Rule 1: Labelled Hydrocephalus but findings confirm widened sulci ─
        if "hydrocephalus" in label.lower():
            wide  = any(cls._keyword_active(kw, findings) for kw in cls.WIDENING_KW)
            comp  = any(cls._keyword_active(kw, findings) for kw in cls.COMPRESSION_KW)
            ad_sc = sum(1 for kw in cls.ALZHEIMER_KW if cls._keyword_active(kw, findings))
            hc_sc = sum(1 for kw in cls.HYDRO_KW     if cls._keyword_active(kw, findings))

            if wide and not comp and ad_sc >= hc_sc:
                old = label
                result["result_label"]     = "Alzheimer's Disease (Moderate)"
                result["confidence_score"] = round(max(confidence - 0.12, 0.50), 4)
                result["priority"]         = "MEDIUM"
                notes.append(
                    f"⚠️ Rule correction: '{old}' → 'Alzheimer's Disease (Moderate)'. "
                    "Findings confirm WIDENED sulci (not compressed) — "
                    "hallmark of Alzheimer's, not hydrocephalus."
                )

        # ── Rule 2: Labelled Alzheimer's but findings confirm compressed sulci ─
        elif "alzheimer" in label.lower():
            comp     = any(cls._keyword_active(kw, findings) for kw in cls.COMPRESSION_KW)
            perivent = cls._keyword_active("periventricular", findings)
            hc_sc    = sum(1 for kw in cls.HYDRO_KW     if cls._keyword_active(kw, findings))
            ad_sc    = sum(1 for kw in cls.ALZHEIMER_KW  if cls._keyword_active(kw, findings))

            if comp and (perivent or hc_sc > ad_sc):
                old = label
                result["result_label"]     = "Hydrocephalus (Normal Pressure / NPH)"
                result["confidence_score"] = round(max(confidence - 0.12, 0.50), 4)
                result["priority"]         = "HIGH"
                notes.append(
                    f"⚠️ Rule correction: '{old}' → 'Hydrocephalus (Normal Pressure / NPH)'. "
                    "Findings confirm COMPRESSED sulci ± periventricular signal — "
                    "hallmark of hydrocephalus, not Alzheimer's."
                )

        # ── Rule 3: Uncertainty language → cap confidence ─────────────────────
        uncertainty_words = (
            "unable", "uncertain", "cannot determine",
            "unclear", "difficult to assess", "limited quality",
        )
        if any(w in findings for w in uncertainty_words):
            old_conf = float(result.get("confidence_score", confidence))
            result["confidence_score"] = round(min(old_conf, 0.55), 4)
            if result["confidence_score"] < old_conf:
                notes.append(
                    "Confidence capped at 0.55 — uncertainty language detected in findings."
                )

        # ── Rule 4: Ventricular enlargement at borderline confidence ──────────
        final_label = result.get("result_label", "")
        final_conf  = float(result.get("confidence_score", 0.75))
        if (
            "ventricular" in findings
            and final_conf < DIFFERENTIAL_THRESHOLD
            and ("alzheimer" in final_label.lower() or "hydrocephalus" in final_label.lower())
        ):
            notes.append(
                "ℹ️ Ventricular enlargement noted at moderate confidence. "
                "Key discriminators: "
                "(1) sulci WIDENED → Alzheimer's; "
                "(2) sulci COMPRESSED + periventricular edema → Hydrocephalus; "
                "(3) hippocampal atrophy → Alzheimer's. "
                "Clinical correlation and volumetric MRI recommended."
            )

        result["validation_notes"] = " | ".join(notes)
        result["disclaimer"]       = SAFETY_DISCLAIMER
        return result


# ─────────────────────────────────────────────────────────────────────────────
# DICOM / NIfTI converters
# ─────────────────────────────────────────────────────────────────────────────

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

        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn) * 255.0
        arr = arr.astype(np.uint8)

        img = Image.fromarray(arr).convert("RGB") if arr.ndim == 2 else Image.fromarray(arr)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name, "PNG")
        logger.info(f"DICOM → PNG: {tmp.name}")
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

        arr    = arr.astype(float)
        lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
        if hi > lo:
            arr = np.clip((arr - lo) / (hi - lo) * 255, 0, 255)
        arr = arr.astype(np.uint8)

        img_pil = Image.fromarray(arr).convert("RGB")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img_pil.save(tmp.name, "PNG")
        logger.info(f"NIfTI → PNG: {tmp.name}")
        return tmp.name

    except ImportError:
        raise RuntimeError("nibabel not installed. Run: pip install nibabel numpy pillow")
    except Exception as e:
        raise RuntimeError(f"NIfTI conversion failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Priority assignment
# ─────────────────────────────────────────────────────────────────────────────

def _assign_priority(label: str, confidence: float) -> str:
    if label in NORMAL_LABELS and confidence >= 0.70:
        return "LOW"
    if label in HIGH_PRIORITY_LABELS and confidence >= 0.50:
        return "HIGH"
    if "alzheimer" in label.lower() and "severe" in label.lower():
        return "HIGH"
    if "hydrocephalus" in label.lower() and "obstructive" in label.lower():
        return "HIGH"
    if label in NORMAL_LABELS:
        return "LOW"
    return "MEDIUM"


def _encode_image_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(scan_type: str) -> str:
    m = MODALITY_PROMPTS.get(scan_type, MODALITY_PROMPTS["general"])
    classes_str = "\n".join(f"  - {c}" for c in m["classes"])

    neuro_rules = ""
    if scan_type in ("brain_mri", "ct_scan"):
        neuro_rules = (
            "\n\nMANDATORY NEURO CHECK before choosing a label:\n"
            "  Step 1 — Are the sulci WIDENED or COMPRESSED?\n"
            "           WIDENED   → favours Alzheimer's Disease\n"
            "           COMPRESSED → favours Hydrocephalus\n"
            "  Step 2 — Is there periventricular T2/FLAIR signal change?\n"
            "           YES → strongly favours Hydrocephalus\n"
            "  Step 3 — Is there hippocampal atrophy?\n"
            "           YES → strongly favours Alzheimer's Disease\n"
            "  Step 4 — Is ventricular enlargement proportional to cortical loss?\n"
            "           YES (ex vacuo) → Alzheimer's\n"
            "           NO (disproportionate) → Hydrocephalus\n"
            "Do NOT skip these steps for any brain scan.\n"
        )

    return (
        "You are a board-certified radiologist AI with 20 years of experience.\n"
        f"You are analyzing a {m['context']} image.\n\n"
        "CRITICAL RULES:\n"
        "1. Choose a LABEL from the provided list ONLY.\n"
        "2. CONFIDENCE must reflect objective image evidence "
        "(clear, unambiguous findings = 0.80–0.95).\n"
        "3. Do NOT default to 'No Abnormality Detected' unless the image is genuinely clear.\n"
        "4. Do NOT report a finding without clear visual evidence.\n"
        "5. If you are uncertain, set CONFIDENCE below 0.60 and say so in FINDINGS.\n"
        "6. If CONFIDENCE is below 0.70, you MUST provide a DIFFERENTIAL.\n"
        + neuro_rules
        + f"\nLook for: {m['focus']}\n\n"
        "Before concluding, assess:\n"
        "- Density/signal changes (hyperdense, hypodense, enhancement)\n"
        "- Structural abnormalities (mass, nodule, consolidation, effusion)\n"
        "- Sulcal pattern: WIDENED (atrophy) vs COMPRESSED (pressure)\n"
        "- Ventricular morphology: rounded under pressure vs proportional to atrophy\n"
        "- Hippocampal volume and signal (brain scans)\n"
        "- Periventricular signal changes (T2/FLAIR hyperintensity)\n"
        "- Symmetry and anatomical variants\n"
        "- Size, borders, surrounding tissue involvement\n"
        "- Overall image quality — if poor, say so and reduce confidence\n\n"
        "Reply in EXACTLY this 9-line format, no extra text:\n"
        "LABEL: <exact diagnosis from list below>\n"
        "CONFIDENCE: <0.00 to 1.00>\n"
        "PRIORITY: <HIGH or MEDIUM or LOW>\n"
        "FINDINGS: <4-5 sentences: what you see, where, sulcal pattern if brain, why this label>\n"
        "CONDITION: <specific disease name>\n"
        "SEVERITY: <Mild or Moderate or Severe or Critical or Early or Advanced or Unable to determine>\n"
        "TREATMENT: <first-line treatment with drug names, dosages, follow-up>\n"
        "DIFFERENTIAL: <if CONFIDENCE <0.70: top 2 alternative diagnoses with one-line reasoning each; else write: None>\n"
        "EXPLANATION: <which specific visual features drove this prediction — sulci, hippocampus, ventricles, lesion location, signal change>\n\n"
        f"LABEL must be one of:\n{classes_str}\n\n"
        "Priority guide: HIGH = malignant / emergency / obstructive. "
        "MEDIUM = needs treatment. LOW = normal / minor finding."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Result extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_result(text: str, model: str) -> dict:
    def get_field(key: str, default: str = "") -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith(key.upper() + ":"):
                val = stripped[len(key) + 1:].strip()
                if val:
                    return val
        return default

    if "LABEL:" in text.upper() or "FINDINGS:" in text.upper():
        label_v  = get_field("LABEL",        "Other Abnormality")
        conf_raw = get_field("CONFIDENCE",   "0.75")
        prio_v   = get_field("PRIORITY",     "MEDIUM").upper()
        find_v   = get_field("FINDINGS",     "")
        cond_v   = get_field("CONDITION",    label_v)
        sev_v    = get_field("SEVERITY",     "Unable to determine")
        treat_v  = get_field("TREATMENT",    "")
        diff_v   = get_field("DIFFERENTIAL", "None")
        expl_v   = get_field("EXPLANATION",  "")

        if prio_v not in ("HIGH", "MEDIUM", "LOW"):
            prio_v = "MEDIUM"

        try:
            conf_v = max(0.0, min(1.0, float(conf_raw)))
        except Exception:
            conf_v = 0.75

        logger.info(f"Parsed [{model}]: {label_v} ({conf_v:.0%})")
        return {
            "result_label":     label_v,
            "confidence_score": conf_v,
            "priority":         prio_v,
            "raw_scores":       {label_v: conf_v},
            "findings":         find_v,
            "cancer_type":      cond_v,
            "stage_estimate":   sev_v,
            "recommendation":   treat_v,
            "differential":     diff_v,
            "explanation":      expl_v,
        }

    # JSON fallback
    try:
        clean = text
        if "```" in clean:
            parts = clean.split("```")
            clean = parts[1] if len(parts) > 1 else clean
            if clean.lower().startswith("json"):
                clean = clean[4:]
        clean = clean.strip()
        s, e = clean.find("{"), clean.rfind("}")
        if s != -1 and e != -1:
            clean = clean[s:e + 1]
        return json.loads(clean)
    except Exception as ex:
        logger.warning(f"Both parsers failed [{model}]: {ex}")
        return {
            "result_label":     "Other Abnormality",
            "confidence_score": 0.50,
            "priority":         "MEDIUM",
            "raw_scores":       {"Other Abnormality": 0.50},
            "findings":         text[:300] if text else "Unable to parse AI response.",
            "cancer_type":      "Unknown",
            "stage_estimate":   "Unable to determine",
            "recommendation":   "Please consult a specialist for review.",
            "differential":     "Unable to determine",
            "explanation":      "",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Vision API call
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini_vision(file_path: str, ext: str, scan_type: str = "general") -> dict:
    api_keys = [k for k in [
        os.environ.get("GEMINI_API_KEY",   ""),
        os.environ.get("GEMINI_API_KEY_2", ""),
        os.environ.get("GEMINI_API_KEY_3", ""),
    ] if k]

    if not api_keys:
        raise RuntimeError("No GEMINI_API_KEY set in .env")

    prompt    = _build_prompt(scan_type)
    image_b64 = _encode_image_base64(file_path)
    mime_type = "image/jpeg" if ext.lower() in ("jpg", "jpeg") else "image/png"

    payload = json.dumps({
        "contents": [{"parts": [
            {"inline_data": {"mime_type": mime_type, "data": image_b64}},
            {"text": prompt},
        ]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1280},
    }).encode("utf-8")

    last_error = "No models tried"

    for api_key in api_keys:
        for model in GEMINI_MODELS:
            url = f"{GEMINI_API_BASE}/{model}:generateContent?key={api_key}"
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw  = json.loads(resp.read().decode("utf-8"))
                text   = raw["candidates"][0]["content"]["parts"][0]["text"].strip()
                result = _extract_result(text, model)

                if result["result_label"] in TREATMENT_MAP and not result.get("recommendation"):
                    result["recommendation"] = TREATMENT_MAP[result["result_label"]]

                return result

            except urllib.error.HTTPError as e:
                body       = e.read().decode("utf-8")
                last_error = f"{model} HTTP {e.code}: {body[:200]}"
                if e.code in (429, 404):
                    continue
                raise RuntimeError(last_error)
            except Exception as ex:
                last_error = f"{model}: {ex}"
                continue

    raise RuntimeError(f"All Gemini models exhausted. Last error: {last_error}")


# ─────────────────────────────────────────────────────────────────────────────
# Simulation fallback
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_simulation(ext: str, error_msg: str = "") -> dict:
    api_key_set = any([
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("GEMINI_API_KEY_2"),
        os.environ.get("GEMINI_API_KEY_3"),
    ])
    if not api_key_set:
        findings = "GEMINI_API_KEY is not set in your .env file."
        rec      = "Get a free key at https://aistudio.google.com"
    else:
        findings = f"Gemini API call failed: {error_msg[:300]}"
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
        "differential":     "Not available",
        "explanation":      "",
        "validation_notes": "",
        "disclaimer":       SAFETY_DISCLAIMER,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main inference entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    file_path:  str,
    upload_id:  int,
    ext:        str,
    model_name: str = "gemini-vision",
    scan_type:  str = "general",
) -> Prediction:
    logger.info(f"run_inference: {file_path} | ext={ext} | scan={scan_type}")

    ext_clean    = ext.lower().lstrip(".")
    tmp_png_path = None

    if ext_clean not in VISION_SUPPORTED:
        result = _fallback_simulation(ext_clean, f"Unsupported file type: {ext_clean}")

    elif not any([
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("GEMINI_API_KEY_2"),
        os.environ.get("GEMINI_API_KEY_3"),
    ]):
        result = _fallback_simulation(ext_clean)

    else:
        try:
            if ext_clean == "dcm":
                tmp_png_path = _dicom_to_png(file_path)
                vision_path  = tmp_png_path
                vision_ext   = "png"
                shutil.copy2(tmp_png_path, file_path.rsplit(".", 1)[0] + "_preview.png")

            elif ext_clean in ("nii", "gz", "nii.gz"):
                tmp_png_path = _nifti_to_png(file_path)
                vision_path  = tmp_png_path
                vision_ext   = "png"
                shutil.copy2(tmp_png_path, file_path.rsplit(".", 1)[0] + "_preview.png")

            else:
                vision_path = file_path
                vision_ext  = ext_clean

            result = _call_gemini_vision(vision_path, vision_ext, scan_type)
            result = NeuroimagingValidator.validate(result, scan_type)

        except Exception as exc:
            logger.error(f"Inference failed: {exc}", exc_info=True)
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
    condition      = result.get("cancer_type", label)
    severity       = result.get("stage_estimate", "Unable to determine")
    recommendation = result.get("recommendation", "")
    differential   = result.get("differential", "None")
    explanation    = result.get("explanation", "")
    val_notes      = result.get("validation_notes", "")
    disclaimer     = result.get("disclaimer", SAFETY_DISCLAIMER)

    if label in TREATMENT_MAP and (not recommendation or len(recommendation) < 30):
        recommendation = TREATMENT_MAP[label]

    raw_scores = {
        k: round(float(v), 4)
        for k, v in result.get("raw_scores", {}).items()
        if isinstance(v, (int, float))
    }

    logger.info(
        f"Final: {label} ({confidence:.0%}) | "
        f"Priority: {priority} | "
        f"Corrected: {bool(val_notes)}"
    )

    return Prediction(
        upload_id        = upload_id,
        result_label     = label,
        confidence_score = round(confidence, 4),
        raw_scores       = json.dumps(raw_scores),
        priority         = priority,
        findings         = findings,
        condition        = condition,
        severity         = severity,
        recommendation   = recommendation,
        differential     = differential,
        explanation      = explanation,
        heatmap_path     = None,
        validation_notes = val_notes,
        disclaimer       = disclaimer,
        analysis_date    = datetime.utcnow(),
        model_version    = "gemini-vision-v2.2",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch inference
# ─────────────────────────────────────────────────────────────────────────────

def batch_infer(
    file_paths: list,
    model_name: str = "gemini-vision",
    scan_type:  str = "general",
) -> list:
    results = []
    for fp in file_paths:
        ext = fp.rsplit(".", 1)[-1].lower().lstrip(".")
        try:
            if ext in VISION_SUPPORTED:
                r = _call_gemini_vision(fp, ext, scan_type)
                r = NeuroimagingValidator.validate(r, scan_type)
            else:
                r = _fallback_simulation(ext)
            results.append((
                r["result_label"],
                float(r["confidence_score"]),
                r.get("raw_scores", {}),
                r.get("differential", "None"),
            ))
        except Exception as exc:
            logger.error(f"Batch error for {fp}: {exc}")
            results.append(("Error", 0.0, {}, ""))
    return results