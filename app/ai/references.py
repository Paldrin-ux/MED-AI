"""
app/ai/references.py  —  Clinical Reference System  (v1.0)
===========================================================
Covers every label used in predict.py across all 8 scan types:
  brain_mri · chest_xray · ct_scan · mri_spine
  abdominal · skin · xray_bone · general

Sources used (all publicly accessible, no API key required):
  Radiopaedia      — radiology imaging gold standard
  DermNet NZ       — skin conditions
  AAD              — American Academy of Dermatology
  AAOS             — American Academy of Orthopaedic Surgeons
  WHO              — infectious diseases, global guidelines
  CDC              — US public health guidelines
  NIH NCI          — cancer types and treatments
  NIH MedlinePlus  — patient-friendly condition overviews
  NIH NINDS        — neurological disorders
  NIH NHLBI        — heart and lung conditions
  NIH NIA          — aging and Alzheimer's
  AHA / ASA        — stroke and cardiac
  Mayo Clinic      — comprehensive condition summaries
  Alzheimer's Assoc— dementia care and staging
  MS Society       — multiple sclerosis
  PubMed           — peer-reviewed evidence links

Usage:
    from app.ai.references import get_references
    refs = get_references("Acne Vulgaris (Moderate)", max_refs=4)
    # returns list of dicts: title, url, source, color
"""

from __future__ import annotations

RefEntry = tuple[str, str, str]   # (title, url, source_tag)

# ─────────────────────────────────────────────────────────────────────────────
# Source badge colours
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_COLORS: dict[str, str] = {
    "Radiopaedia":       "#3b82f6",
    "AAD":               "#8b5cf6",
    "DermNet NZ":        "#ec4899",
    "AAOS":              "#64748b",
    "NIH NCI":           "#0ea5e9",
    "NIH MedlinePlus":   "#06b6d4",
    "NIH NINDS":         "#0891b2",
    "NIH NHLBI":         "#0284c7",
    "NIH NIA":           "#0369a1",
    "NIH":               "#0ea5e9",
    "WHO":               "#16a34a",
    "CDC":               "#dc2626",
    "AHA":               "#ef4444",
    "AHA/ASA":           "#ef4444",
    "Mayo Clinic":       "#f59e0b",
    "PubMed":            "#10b981",
    "MS Society":        "#7c3aed",
    "Alzheimer's Assoc": "#6366f1",
}

# ─────────────────────────────────────────────────────────────────────────────
# Master reference map  —  all 121 labels
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE_MAP: dict[str, list[RefEntry]] = {

    # ══════════════════════════════════════════════════════════════════════════
    # BRAIN MRI  (16 labels)
    # ══════════════════════════════════════════════════════════════════════════

    "Alzheimer's Disease (Mild)": [
        ("Radiopaedia — Alzheimer disease",         "https://radiopaedia.org/articles/alzheimer-disease", "Radiopaedia"),
        ("Alzheimer's Association — Diagnosis",     "https://www.alz.org/alzheimers-dementia/diagnosis", "Alzheimer's Assoc"),
        ("NIH NIA — Alzheimer's overview",          "https://www.nia.nih.gov/health/alzheimers-and-dementia", "NIH NIA"),
        ("PubMed — Donepezil mild AD",              "https://pubmed.ncbi.nlm.nih.gov/?term=alzheimer+mild+donepezil+cholinesterase", "PubMed"),
    ],
    "Alzheimer's Disease (Moderate)": [
        ("Radiopaedia — Alzheimer disease",         "https://radiopaedia.org/articles/alzheimer-disease", "Radiopaedia"),
        ("Alzheimer's Association — Stages",        "https://www.alz.org/alzheimers-dementia/stages", "Alzheimer's Assoc"),
        ("NIH NIA — Treatment options",             "https://www.nia.nih.gov/health/alzheimers-causes-and-risk-factors", "NIH NIA"),
        ("PubMed — Memantine moderate AD",          "https://pubmed.ncbi.nlm.nih.gov/?term=moderate+alzheimer+memantine+donepezil", "PubMed"),
    ],
    "Alzheimer's Disease (Severe)": [
        ("Radiopaedia — Alzheimer disease",         "https://radiopaedia.org/articles/alzheimer-disease", "Radiopaedia"),
        ("Alzheimer's Assoc — Late stage care",     "https://www.alz.org/help-support/caregiving/stages-behaviors/late-stage", "Alzheimer's Assoc"),
        ("NIH MedlinePlus — Alzheimer care",        "https://medlineplus.gov/alzheimersdisease.html", "NIH MedlinePlus"),
        ("Mayo Clinic — Late-stage Alzheimer's",    "https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/in-depth/alzheimers-stages/art-20048448", "Mayo Clinic"),
    ],
    "Hydrocephalus (Normal Pressure / NPH)": [
        ("Radiopaedia — Normal pressure hydrocephalus", "https://radiopaedia.org/articles/normal-pressure-hydrocephalus", "Radiopaedia"),
        ("NIH MedlinePlus — Hydrocephalus",         "https://medlineplus.gov/hydrocephalus.html", "NIH MedlinePlus"),
        ("Mayo Clinic — Hydrocephalus",             "https://www.mayoclinic.org/diseases-conditions/hydrocephalus/symptoms-causes/syc-20373604", "Mayo Clinic"),
        ("PubMed — NPH shunt tap test",             "https://pubmed.ncbi.nlm.nih.gov/?term=normal+pressure+hydrocephalus+shunt+tap+test", "PubMed"),
    ],
    "Hydrocephalus (Obstructive)": [
        ("Radiopaedia — Obstructive hydrocephalus", "https://radiopaedia.org/articles/obstructive-hydrocephalus", "Radiopaedia"),
        ("NIH MedlinePlus — Hydrocephalus",         "https://medlineplus.gov/hydrocephalus.html", "NIH MedlinePlus"),
        ("PubMed — ETV obstructive hydrocephalus",  "https://pubmed.ncbi.nlm.nih.gov/?term=obstructive+hydrocephalus+endoscopic+third+ventriculostomy", "PubMed"),
    ],
    "Glioblastoma (GBM)": [
        ("Radiopaedia — Glioblastoma",              "https://radiopaedia.org/articles/glioblastoma-1", "Radiopaedia"),
        ("NIH NCI — Glioblastoma",                  "https://www.cancer.gov/types/brain/patient/adult-brain-treatment-pdq", "NIH NCI"),
        ("Mayo Clinic — Glioblastoma",              "https://www.mayoclinic.org/diseases-conditions/glioblastoma/cdc-20350148", "Mayo Clinic"),
        ("PubMed — GBM temozolomide radiation",     "https://pubmed.ncbi.nlm.nih.gov/?term=glioblastoma+temozolomide+radiotherapy+MGMT", "PubMed"),
    ],
    "Astrocytoma (Grade II-III)": [
        ("Radiopaedia — Astrocytoma",               "https://radiopaedia.org/articles/astrocytoma", "Radiopaedia"),
        ("NIH NCI — Brain tumors",                  "https://www.cancer.gov/types/brain", "NIH NCI"),
        ("PubMed — IDH astrocytoma grading",        "https://pubmed.ncbi.nlm.nih.gov/?term=astrocytoma+IDH+grade+MRI+imaging", "PubMed"),
    ],
    "Meningioma": [
        ("Radiopaedia — Meningioma",                "https://radiopaedia.org/articles/meningioma", "Radiopaedia"),
        ("NIH NCI — Meningioma",                    "https://www.cancer.gov/types/brain/patient/adult-brain-treatment-pdq", "NIH NCI"),
        ("Mayo Clinic — Meningioma",                "https://www.mayoclinic.org/diseases-conditions/meningioma/symptoms-causes/syc-20355643", "Mayo Clinic"),
    ],
    "Pituitary Adenoma": [
        ("Radiopaedia — Pituitary adenoma",         "https://radiopaedia.org/articles/pituitary-adenoma", "Radiopaedia"),
        ("NIH MedlinePlus — Pituitary tumors",      "https://medlineplus.gov/pituitarytumors.html", "NIH MedlinePlus"),
        ("PubMed — Pituitary adenoma MRI cabergoline", "https://pubmed.ncbi.nlm.nih.gov/?term=pituitary+adenoma+MRI+cabergoline+transsphenoidal", "PubMed"),
    ],
    "Brain Metastasis": [
        ("Radiopaedia — Brain metastases",          "https://radiopaedia.org/articles/brain-metastases", "Radiopaedia"),
        ("NIH NCI — Brain metastases",              "https://www.cancer.gov/types/metastatic-cancer", "NIH NCI"),
        ("PubMed — Brain metastasis SRS WBRT",      "https://pubmed.ncbi.nlm.nih.gov/?term=brain+metastasis+stereotactic+radiosurgery+WBRT", "PubMed"),
    ],
    "Ischemic Stroke": [
        ("Radiopaedia — Ischaemic stroke",          "https://radiopaedia.org/articles/ischaemic-stroke", "Radiopaedia"),
        ("AHA/ASA — Ischemic stroke guidelines",    "https://www.stroke.org/en/about-stroke/types-of-stroke/ischemic-stroke-clots", "AHA/ASA"),
        ("NIH MedlinePlus — Stroke",                "https://medlineplus.gov/stroke.html", "NIH MedlinePlus"),
        ("PubMed — tPA thrombectomy stroke",        "https://pubmed.ncbi.nlm.nih.gov/?term=ischemic+stroke+tPA+thrombectomy+outcome", "PubMed"),
    ],
    "Hemorrhagic Stroke": [
        ("Radiopaedia — Intracerebral haemorrhage", "https://radiopaedia.org/articles/intracerebral-haemorrhage", "Radiopaedia"),
        ("AHA/ASA — Hemorrhagic stroke",            "https://www.stroke.org/en/about-stroke/types-of-stroke/hemorrhagic-strokes-bleeds", "AHA/ASA"),
        ("NIH MedlinePlus — Hemorrhagic stroke",    "https://medlineplus.gov/hemorrhagicstroke.html", "NIH MedlinePlus"),
    ],
    "Multiple Sclerosis (MS)": [
        ("Radiopaedia — Multiple sclerosis",        "https://radiopaedia.org/articles/multiple-sclerosis", "Radiopaedia"),
        ("NIH NINDS — MS overview",                 "https://www.ninds.nih.gov/health-information/disorders/multiple-sclerosis", "NIH NINDS"),
        ("National MS Society",                     "https://www.nationalmssociety.org/What-is-MS", "MS Society"),
        ("PubMed — MS McDonald MRI criteria",       "https://pubmed.ncbi.nlm.nih.gov/?term=multiple+sclerosis+MRI+McDonald+criteria+diagnosis", "PubMed"),
    ],
    "Brain Abscess": [
        ("Radiopaedia — Cerebral abscess",          "https://radiopaedia.org/articles/cerebral-abscess", "Radiopaedia"),
        ("NIH MedlinePlus — Brain abscess",         "https://medlineplus.gov/ency/article/000783.htm", "NIH MedlinePlus"),
        ("PubMed — Brain abscess ceftriaxone metronidazole", "https://pubmed.ncbi.nlm.nih.gov/?term=brain+abscess+ceftriaxone+metronidazole+drainage", "PubMed"),
    ],

    # ══════════════════════════════════════════════════════════════════════════
    # CHEST X-RAY  (17 labels)
    # ══════════════════════════════════════════════════════════════════════════

    "Tuberculosis (TB)": [
        ("WHO — TB treatment guidelines",           "https://www.who.int/tb/areas-of-work/treatment/en/", "WHO"),
        ("Radiopaedia — Pulmonary tuberculosis",    "https://radiopaedia.org/articles/pulmonary-tuberculosis", "Radiopaedia"),
        ("CDC — TB treatment",                      "https://www.cdc.gov/tb/topic/treatment/tbdisease.htm", "CDC"),
        ("PubMed — TB RIPE regimen outcomes",       "https://pubmed.ncbi.nlm.nih.gov/?term=tuberculosis+RIPE+treatment+outcomes+isoniazid", "PubMed"),
    ],
    "Pneumonia (Bacterial)": [
        ("Radiopaedia — Bacterial pneumonia",       "https://radiopaedia.org/articles/bacterial-pneumonia", "Radiopaedia"),
        ("NIH MedlinePlus — Pneumonia",             "https://medlineplus.gov/pneumonia.html", "NIH MedlinePlus"),
        ("Mayo Clinic — Pneumonia",                 "https://www.mayoclinic.org/diseases-conditions/pneumonia/symptoms-causes/syc-20354204", "Mayo Clinic"),
        ("PubMed — CAP antibiotic guidelines",      "https://pubmed.ncbi.nlm.nih.gov/?term=community+acquired+pneumonia+antibiotic+guidelines+ceftriaxone", "PubMed"),
    ],
    "Pneumonia (Viral)": [
        ("Radiopaedia — Viral pneumonia",           "https://radiopaedia.org/articles/viral-pneumonia", "Radiopaedia"),
        ("CDC — Influenza antiviral treatment",     "https://www.cdc.gov/flu/treatment/index.html", "CDC"),
        ("NIH MedlinePlus — Viral pneumonia",       "https://medlineplus.gov/pneumonia.html", "NIH MedlinePlus"),
    ],
    "COVID-19": [
        ("Radiopaedia — COVID-19 pneumonia",        "https://radiopaedia.org/articles/coronavirus-disease-2019-covid-19-3", "Radiopaedia"),
        ("WHO — COVID-19 therapeutics",             "https://www.who.int/publications/i/item/WHO-2019-nCoV-therapeutics-2022.5", "WHO"),
        ("NIH — COVID-19 treatment guidelines",     "https://www.covid19treatmentguidelines.nih.gov/", "NIH"),
    ],
    "Lung Adenocarcinoma": [
        ("Radiopaedia — Lung adenocarcinoma",       "https://radiopaedia.org/articles/pulmonary-adenocarcinoma", "Radiopaedia"),
        ("NIH NCI — Lung cancer treatment",         "https://www.cancer.gov/types/lung", "NIH NCI"),
        ("Mayo Clinic — Lung cancer",               "https://www.mayoclinic.org/diseases-conditions/lung-cancer/symptoms-causes/syc-20374620", "Mayo Clinic"),
        ("PubMed — EGFR osimertinib adenocarcinoma","https://pubmed.ncbi.nlm.nih.gov/?term=lung+adenocarcinoma+EGFR+osimertinib+targeted", "PubMed"),
    ],
    "Squamous Cell Carcinoma": [
        ("Radiopaedia — Squamous cell lung cancer", "https://radiopaedia.org/articles/squamous-cell-carcinoma-of-the-lung", "Radiopaedia"),
        ("NIH NCI — Lung cancer types",             "https://www.cancer.gov/types/lung/patient/lung-treatment-pdq", "NIH NCI"),
        ("PubMed — SCC lung immunotherapy",         "https://pubmed.ncbi.nlm.nih.gov/?term=squamous+cell+lung+carcinoma+immunotherapy+pembrolizumab", "PubMed"),
    ],
    "Small Cell Lung Cancer": [
        ("Radiopaedia — Small cell lung cancer",    "https://radiopaedia.org/articles/small-cell-lung-cancer", "Radiopaedia"),
        ("NIH NCI — Small cell lung cancer",        "https://www.cancer.gov/types/lung/patient/small-cell-lung-treatment-pdq", "NIH NCI"),
        ("PubMed — SCLC etoposide cisplatin",       "https://pubmed.ncbi.nlm.nih.gov/?term=small+cell+lung+cancer+etoposide+cisplatin+atezolizumab", "PubMed"),
    ],
    "Pulmonary Metastasis": [
        ("Radiopaedia — Pulmonary metastases",      "https://radiopaedia.org/articles/pulmonary-metastases", "Radiopaedia"),
        ("NIH NCI — Metastatic cancer",             "https://www.cancer.gov/types/metastatic-cancer", "NIH NCI"),
        ("NIH MedlinePlus — Lung metastases",       "https://medlineplus.gov/ency/article/000097.htm", "NIH MedlinePlus"),
    ],
    "Pleural Effusion": [
        ("Radiopaedia — Pleural effusion",          "https://radiopaedia.org/articles/pleural-effusion", "Radiopaedia"),
        ("NIH MedlinePlus — Pleural disorders",     "https://medlineplus.gov/pleuraldisorders.html", "NIH MedlinePlus"),
        ("PubMed — Thoracentesis diagnosis",        "https://pubmed.ncbi.nlm.nih.gov/?term=pleural+effusion+thoracentesis+diagnosis+management", "PubMed"),
    ],
    "Pneumothorax": [
        ("Radiopaedia — Pneumothorax",              "https://radiopaedia.org/articles/pneumothorax", "Radiopaedia"),
        ("NIH MedlinePlus — Pneumothorax",          "https://medlineplus.gov/ency/article/000087.htm", "NIH MedlinePlus"),
        ("PubMed — Pneumothorax chest tube",        "https://pubmed.ncbi.nlm.nih.gov/?term=pneumothorax+chest+tube+needle+decompression+management", "PubMed"),
    ],
    "Cardiomegaly": [
        ("Radiopaedia — Cardiomegaly",              "https://radiopaedia.org/articles/cardiomegaly", "Radiopaedia"),
        ("AHA — Heart failure overview",            "https://www.heart.org/en/health-topics/heart-failure", "AHA"),
        ("NIH MedlinePlus — Heart failure",         "https://medlineplus.gov/heartfailure.html", "NIH MedlinePlus"),
    ],
    "Pulmonary Edema": [
        ("Radiopaedia — Pulmonary oedema",          "https://radiopaedia.org/articles/pulmonary-oedema", "Radiopaedia"),
        ("AHA — Acute heart failure",               "https://www.heart.org/en/health-topics/heart-failure/treatment-options-for-heart-failure", "AHA"),
        ("NIH MedlinePlus — Pulmonary edema",       "https://medlineplus.gov/ency/article/000140.htm", "NIH MedlinePlus"),
    ],
    "Pulmonary Fibrosis": [
        ("Radiopaedia — Idiopathic pulmonary fibrosis", "https://radiopaedia.org/articles/idiopathic-pulmonary-fibrosis", "Radiopaedia"),
        ("NIH NHLBI — Pulmonary fibrosis",          "https://www.nhlbi.nih.gov/health/pulmonary-fibrosis", "NIH NHLBI"),
        ("PubMed — Nintedanib pirfenidone IPF",     "https://pubmed.ncbi.nlm.nih.gov/?term=idiopathic+pulmonary+fibrosis+nintedanib+pirfenidone+treatment", "PubMed"),
    ],
    "Atelectasis": [
        ("Radiopaedia — Atelectasis",               "https://radiopaedia.org/articles/atelectasis", "Radiopaedia"),
        ("NIH MedlinePlus — Atelectasis",           "https://medlineplus.gov/ency/article/000065.htm", "NIH MedlinePlus"),
        ("Mayo Clinic — Atelectasis",               "https://www.mayoclinic.org/diseases-conditions/atelectasis/symptoms-causes/syc-20369684", "Mayo Clinic"),
    ],
    "Lymphoma": [
        ("Radiopaedia — Lymphoma chest",            "https://radiopaedia.org/articles/lymphoma-of-the-chest", "Radiopaedia"),
        ("NIH NCI — Lymphoma",                      "https://www.cancer.gov/types/lymphoma", "NIH NCI"),
        ("Mayo Clinic — Lymphoma",                  "https://www.mayoclinic.org/diseases-conditions/lymphoma/symptoms-causes/syc-20352638", "Mayo Clinic"),
        ("PubMed — Lymphoma imaging diagnosis",     "https://pubmed.ncbi.nlm.nih.gov/?term=lymphoma+CT+imaging+diagnosis+staging", "PubMed"),
    ],

    # ══════════════════════════════════════════════════════════════════════════
    # CT SCAN  (16 labels — some shared with brain/abdominal above)
    # ══════════════════════════════════════════════════════════════════════════

    "Glioblastoma / Brain Tumor": [
        ("Radiopaedia — Glioblastoma",              "https://radiopaedia.org/articles/glioblastoma-1", "Radiopaedia"),
        ("NIH NCI — Brain tumors",                  "https://www.cancer.gov/types/brain", "NIH NCI"),
        ("Mayo Clinic — Glioblastoma",              "https://www.mayoclinic.org/diseases-conditions/glioblastoma/cdc-20350148", "Mayo Clinic"),
        ("PubMed — Brain tumor CT MRI features",    "https://pubmed.ncbi.nlm.nih.gov/?term=brain+tumor+CT+MRI+ring+enhancement+diagnosis", "PubMed"),
    ],
    "Lung Cancer": [
        ("Radiopaedia — Lung cancer CT",            "https://radiopaedia.org/articles/lung-cancer", "Radiopaedia"),
        ("NIH NCI — Lung cancer",                   "https://www.cancer.gov/types/lung", "NIH NCI"),
        ("CDC — Lung cancer overview",              "https://www.cdc.gov/cancer/lung/index.htm", "CDC"),
        ("PubMed — Lung cancer CT screening",       "https://pubmed.ncbi.nlm.nih.gov/?term=lung+cancer+CT+low+dose+screening+diagnosis", "PubMed"),
    ],
    "Colorectal Cancer": [
        ("Radiopaedia — Colorectal carcinoma",      "https://radiopaedia.org/articles/colorectal-carcinoma", "Radiopaedia"),
        ("NIH NCI — Colon cancer",                  "https://www.cancer.gov/types/colorectal", "NIH NCI"),
        ("Mayo Clinic — Colon cancer",              "https://www.mayoclinic.org/diseases-conditions/colon-cancer/symptoms-causes/syc-20353669", "Mayo Clinic"),
        ("PubMed — Colorectal cancer CT staging",   "https://pubmed.ncbi.nlm.nih.gov/?term=colorectal+cancer+CT+staging+FOLFOX+treatment", "PubMed"),
    ],
    "Spinal / Bone Metastasis": [
        ("Radiopaedia — Spinal metastases",         "https://radiopaedia.org/articles/spinal-metastases", "Radiopaedia"),
        ("NIH NCI — Bone metastases",               "https://www.cancer.gov/types/metastatic-cancer", "NIH NCI"),
        ("PubMed — Spinal metastasis radiation",    "https://pubmed.ncbi.nlm.nih.gov/?term=spinal+bone+metastasis+radiation+bisphosphonate", "PubMed"),
    ],
    "Aortic Aneurysm": [
        ("Radiopaedia — Aortic aneurysm",           "https://radiopaedia.org/articles/aortic-aneurysm", "Radiopaedia"),
        ("NIH MedlinePlus — Aortic aneurysm",       "https://medlineplus.gov/aorticaneurysm.html", "NIH MedlinePlus"),
        ("AHA — Aortic aneurysm",                   "https://www.heart.org/en/health-topics/aortic-aneurysm", "AHA"),
        ("PubMed — AAA CT screening surgical repair","https://pubmed.ncbi.nlm.nih.gov/?term=aortic+aneurysm+CT+screening+endovascular+repair", "PubMed"),
    ],
    "Appendicitis": [
        ("Radiopaedia — Appendicitis",              "https://radiopaedia.org/articles/appendicitis", "Radiopaedia"),
        ("NIH MedlinePlus — Appendicitis",          "https://medlineplus.gov/appendicitis.html", "NIH MedlinePlus"),
        ("Mayo Clinic — Appendicitis",              "https://www.mayoclinic.org/diseases-conditions/appendicitis/symptoms-causes/syc-20369543", "Mayo Clinic"),
        ("PubMed — Appendicitis CT diagnosis",      "https://pubmed.ncbi.nlm.nih.gov/?term=appendicitis+CT+diagnosis+appendectomy", "PubMed"),
    ],
    "Diverticulitis": [
        ("Radiopaedia — Diverticulitis",            "https://radiopaedia.org/articles/diverticulitis", "Radiopaedia"),
        ("NIH MedlinePlus — Diverticulitis",        "https://medlineplus.gov/diverticulosisanddiverticulitis.html", "NIH MedlinePlus"),
        ("Mayo Clinic — Diverticulitis",            "https://www.mayoclinic.org/diseases-conditions/diverticulitis/symptoms-causes/syc-20371758", "Mayo Clinic"),
        ("PubMed — Diverticulitis CT management",   "https://pubmed.ncbi.nlm.nih.gov/?term=diverticulitis+CT+antibiotic+treatment+management", "PubMed"),
    ],

    # ══════════════════════════════════════════════════════════════════════════
    # SPINE MRI  (11 labels)
    # ══════════════════════════════════════════════════════════════════════════

    "Disc Herniation": [
        ("Radiopaedia — Lumbar disc herniation",    "https://radiopaedia.org/articles/lumbar-disc-herniation", "Radiopaedia"),
        ("AAOS — Herniated disk",                   "https://orthoinfo.aaos.org/en/diseases--conditions/herniated-disk/", "AAOS"),
        ("NIH MedlinePlus — Herniated disk",        "https://medlineplus.gov/herniatedisk.html", "NIH MedlinePlus"),
        ("PubMed — Microdiscectomy outcomes",       "https://pubmed.ncbi.nlm.nih.gov/?term=lumbar+disc+herniation+microdiscectomy+outcomes", "PubMed"),
    ],
    "Spinal Stenosis": [
        ("Radiopaedia — Spinal canal stenosis",     "https://radiopaedia.org/articles/spinal-canal-stenosis", "Radiopaedia"),
        ("AAOS — Lumbar spinal stenosis",           "https://orthoinfo.aaos.org/en/diseases--conditions/lumbar-spinal-stenosis/", "AAOS"),
        ("NIH MedlinePlus — Spinal stenosis",       "https://medlineplus.gov/spinalstenosisinlumbarregion.html", "NIH MedlinePlus"),
    ],
    "Vertebral Metastasis": [
        ("Radiopaedia — Vertebral metastases",      "https://radiopaedia.org/articles/spinal-metastases", "Radiopaedia"),
        ("NIH NCI — Bone metastases",               "https://www.cancer.gov/types/metastatic-cancer", "NIH NCI"),
        ("PubMed — Vertebral metastasis MRI",       "https://pubmed.ncbi.nlm.nih.gov/?term=vertebral+metastasis+MRI+treatment+radiation", "PubMed"),
    ],
    "Spinal Cord Tumor": [
        ("Radiopaedia — Spinal cord tumours",       "https://radiopaedia.org/articles/spinal-cord-tumours", "Radiopaedia"),
        ("NIH NCI — Spinal cord tumors",            "https://www.cancer.gov/types/brain", "NIH NCI"),
        ("NIH NINDS — Spinal cord tumor",           "https://www.ninds.nih.gov/health-information/disorders/spinal-cord-tumors", "NIH NINDS"),
    ],
    "Multiple Myeloma": [
        ("Radiopaedia — Multiple myeloma spine",    "https://radiopaedia.org/articles/multiple-myeloma-spine", "Radiopaedia"),
        ("NIH NCI — Multiple myeloma",              "https://www.cancer.gov/types/myeloma", "NIH NCI"),
        ("Mayo Clinic — Multiple myeloma",          "https://www.mayoclinic.org/diseases-conditions/multiple-myeloma/symptoms-causes/syc-20353378", "Mayo Clinic"),
        ("PubMed — Multiple myeloma bortezomib",    "https://pubmed.ncbi.nlm.nih.gov/?term=multiple+myeloma+bortezomib+lenalidomide+treatment", "PubMed"),
    ],
    "Compression Fracture": [
        ("Radiopaedia — Vertebral compression fracture", "https://radiopaedia.org/articles/vertebral-compression-fracture", "Radiopaedia"),
        ("AAOS — Vertebral fractures",              "https://orthoinfo.aaos.org/en/diseases--conditions/vertebral-fractures/", "AAOS"),
        ("NIH MedlinePlus — Vertebral fracture",    "https://medlineplus.gov/ency/article/007459.htm", "NIH MedlinePlus"),
    ],
    "Spondylolisthesis": [
        ("Radiopaedia — Spondylolisthesis",         "https://radiopaedia.org/articles/spondylolisthesis", "Radiopaedia"),
        ("AAOS — Spondylolysis and spondylolisthesis", "https://orthoinfo.aaos.org/en/diseases--conditions/spondylolysis-and-spondylolisthesis/", "AAOS"),
        ("NIH MedlinePlus — Spondylolisthesis",     "https://medlineplus.gov/ency/article/001260.htm", "NIH MedlinePlus"),
    ],
    "Infection / Discitis": [
        ("Radiopaedia — Discitis osteomyelitis",    "https://radiopaedia.org/articles/discitis-1", "Radiopaedia"),
        ("NIH MedlinePlus — Diskitis",              "https://medlineplus.gov/ency/article/001264.htm", "NIH MedlinePlus"),
        ("PubMed — Spondylodiscitis MRI antibiotics","https://pubmed.ncbi.nlm.nih.gov/?term=spondylodiscitis+MRI+antibiotic+treatment", "PubMed"),
    ],
    "Syringomyelia": [
        ("Radiopaedia — Syringomyelia",             "https://radiopaedia.org/articles/syringohydromyelia", "Radiopaedia"),
        ("NIH NINDS — Syringomyelia",               "https://www.ninds.nih.gov/health-information/disorders/syringomyelia", "NIH NINDS"),
        ("Mayo Clinic — Syringomyelia",             "https://www.mayoclinic.org/diseases-conditions/syringomyelia/symptoms-causes/syc-20377113", "Mayo Clinic"),
    ],

    # ══════════════════════════════════════════════════════════════════════════
    # ABDOMINAL  (13 labels)
    # ══════════════════════════════════════════════════════════════════════════

    "Hepatocellular Carcinoma (HCC)": [
        ("Radiopaedia — Hepatocellular carcinoma",  "https://radiopaedia.org/articles/hepatocellular-carcinoma", "Radiopaedia"),
        ("NIH NCI — Liver cancer",                  "https://www.cancer.gov/types/liver/patient/adult-liver-treatment-pdq", "NIH NCI"),
        ("Mayo Clinic — Liver cancer",              "https://www.mayoclinic.org/diseases-conditions/liver-cancer/symptoms-causes/syc-20353659", "Mayo Clinic"),
        ("PubMed — HCC sorafenib TACE",             "https://pubmed.ncbi.nlm.nih.gov/?term=hepatocellular+carcinoma+sorafenib+TACE+treatment", "PubMed"),
    ],
    "Liver Metastasis": [
        ("Radiopaedia — Liver metastases",          "https://radiopaedia.org/articles/liver-metastases", "Radiopaedia"),
        ("NIH NCI — Liver metastases",              "https://www.cancer.gov/types/metastatic-cancer", "NIH NCI"),
        ("PubMed — Liver metastasis resection",     "https://pubmed.ncbi.nlm.nih.gov/?term=liver+metastasis+resection+chemotherapy+outcomes", "PubMed"),
    ],
    "Renal Cell Carcinoma": [
        ("Radiopaedia — Renal cell carcinoma",      "https://radiopaedia.org/articles/renal-cell-carcinoma", "Radiopaedia"),
        ("NIH NCI — Kidney cancer",                 "https://www.cancer.gov/types/kidney/patient/kidney-treatment-pdq", "NIH NCI"),
        ("PubMed — RCC sunitinib cabozantinib",     "https://pubmed.ncbi.nlm.nih.gov/?term=renal+cell+carcinoma+sunitinib+cabozantinib+treatment", "PubMed"),
    ],
    "Pancreatic Adenocarcinoma": [
        ("Radiopaedia — Pancreatic ductal adenocarcinoma", "https://radiopaedia.org/articles/pancreatic-ductal-adenocarcinoma", "Radiopaedia"),
        ("NIH NCI — Pancreatic cancer",             "https://www.cancer.gov/types/pancreatic/patient/pancreatic-treatment-pdq", "NIH NCI"),
        ("PubMed — Pancreatic cancer FOLFIRINOX",   "https://pubmed.ncbi.nlm.nih.gov/?term=pancreatic+cancer+FOLFIRINOX+gemcitabine+treatment", "PubMed"),
    ],
    "Adrenal Carcinoma": [
        ("Radiopaedia — Adrenal cortical carcinoma","https://radiopaedia.org/articles/adrenocortical-carcinoma", "Radiopaedia"),
        ("NIH NCI — Adrenal cortical carcinoma",    "https://www.cancer.gov/types/adrenocortical", "NIH NCI"),
        ("NIH MedlinePlus — Adrenal gland cancer",  "https://medlineplus.gov/ency/article/001663.htm", "NIH MedlinePlus"),
        ("PubMed — Adrenal carcinoma mitotane",     "https://pubmed.ncbi.nlm.nih.gov/?term=adrenocortical+carcinoma+mitotane+treatment+surgery", "PubMed"),
    ],
    "Colorectal Cancer": [
        ("Radiopaedia — Colorectal carcinoma",      "https://radiopaedia.org/articles/colorectal-carcinoma", "Radiopaedia"),
        ("NIH NCI — Colorectal cancer",             "https://www.cancer.gov/types/colorectal", "NIH NCI"),
        ("Mayo Clinic — Colon cancer",              "https://www.mayoclinic.org/diseases-conditions/colon-cancer/symptoms-causes/syc-20353669", "Mayo Clinic"),
        ("PubMed — Colorectal cancer treatment",    "https://pubmed.ncbi.nlm.nih.gov/?term=colorectal+cancer+FOLFOX+bevacizumab+treatment", "PubMed"),
    ],
    "Peritoneal Carcinomatosis": [
        ("Radiopaedia — Peritoneal carcinomatosis", "https://radiopaedia.org/articles/peritoneal-carcinomatosis", "Radiopaedia"),
        ("NIH NCI — Peritoneal metastases",         "https://www.cancer.gov/types/metastatic-cancer", "NIH NCI"),
        ("PubMed — Peritoneal carcinomatosis HIPEC","https://pubmed.ncbi.nlm.nih.gov/?term=peritoneal+carcinomatosis+HIPEC+cytoreductive+surgery", "PubMed"),
    ],
    "Gallstones (Cholelithiasis)": [
        ("Radiopaedia — Cholelithiasis",            "https://radiopaedia.org/articles/cholelithiasis", "Radiopaedia"),
        ("NIH MedlinePlus — Gallstones",            "https://medlineplus.gov/gallstones.html", "NIH MedlinePlus"),
        ("Mayo Clinic — Gallstones",                "https://www.mayoclinic.org/diseases-conditions/gallstones/symptoms-causes/syc-20354214", "Mayo Clinic"),
    ],
    "Liver Cirrhosis": [
        ("Radiopaedia — Liver cirrhosis",           "https://radiopaedia.org/articles/liver-cirrhosis", "Radiopaedia"),
        ("NIH MedlinePlus — Cirrhosis",             "https://medlineplus.gov/cirrhosis.html", "NIH MedlinePlus"),
        ("Mayo Clinic — Cirrhosis",                 "https://www.mayoclinic.org/diseases-conditions/cirrhosis/symptoms-causes/syc-20351487", "Mayo Clinic"),
        ("PubMed — Cirrhosis portal hypertension beta-blocker", "https://pubmed.ncbi.nlm.nih.gov/?term=liver+cirrhosis+propranolol+portal+hypertension", "PubMed"),
    ],
    "Splenomegaly": [
        ("Radiopaedia — Splenomegaly",              "https://radiopaedia.org/articles/splenomegaly", "Radiopaedia"),
        ("NIH MedlinePlus — Enlarged spleen",       "https://medlineplus.gov/ency/article/003276.htm", "NIH MedlinePlus"),
        ("Mayo Clinic — Enlarged spleen",           "https://www.mayoclinic.org/diseases-conditions/enlarged-spleen/symptoms-causes/syc-20354326", "Mayo Clinic"),
    ],
    "Kidney Stone (Nephrolithiasis)": [
        ("Radiopaedia — Nephrolithiasis",           "https://radiopaedia.org/articles/nephrolithiasis", "Radiopaedia"),
        ("NIH MedlinePlus — Kidney stones",         "https://medlineplus.gov/kidneystones.html", "NIH MedlinePlus"),
        ("Mayo Clinic — Kidney stones",             "https://www.mayoclinic.org/diseases-conditions/kidney-stones/symptoms-causes/syc-20355755", "Mayo Clinic"),
    ],
    "Benign Finding": [
        ("Radiopaedia — Benign abdominal lesions",  "https://radiopaedia.org/articles/benign-hepatic-lesions", "Radiopaedia"),
        ("NIH MedlinePlus — Benign tumors",         "https://medlineplus.gov/benigntumors.html", "NIH MedlinePlus"),
    ],

    # ══════════════════════════════════════════════════════════════════════════
    # SKIN / DERMATOLOGY  (21 labels)
    # ══════════════════════════════════════════════════════════════════════════

    "Healthy Skin": [
        ("AAD — Healthy skin tips",                 "https://www.aad.org/public/everyday-care/skin-care-basics/care/skin-care-routine", "AAD"),
        ("DermNet NZ — Normal skin",                "https://dermnetnz.org/topics/normal-skin", "DermNet NZ"),
    ],
    "Acne Vulgaris (Mild)": [
        ("AAD — Acne overview",                     "https://www.aad.org/public/diseases/acne", "AAD"),
        ("DermNet NZ — Acne vulgaris",              "https://dermnetnz.org/topics/acne", "DermNet NZ"),
        ("NIH MedlinePlus — Acne",                  "https://medlineplus.gov/acne.html", "NIH MedlinePlus"),
        ("PubMed — Topical retinoid benzoyl peroxide acne", "https://pubmed.ncbi.nlm.nih.gov/?term=acne+vulgaris+topical+retinoid+benzoyl+peroxide+mild", "PubMed"),
    ],
    "Acne Vulgaris (Moderate)": [
        ("AAD — Acne treatment",                    "https://www.aad.org/public/diseases/acne/derm-treat/treat-acne", "AAD"),
        ("DermNet NZ — Acne vulgaris",              "https://dermnetnz.org/topics/acne", "DermNet NZ"),
        ("NIH MedlinePlus — Acne",                  "https://medlineplus.gov/acne.html", "NIH MedlinePlus"),
        ("PubMed — Doxycycline moderate acne",      "https://pubmed.ncbi.nlm.nih.gov/?term=moderate+acne+doxycycline+oral+antibiotic+treatment", "PubMed"),
    ],
    "Acne Vulgaris (Severe/Cystic)": [
        ("AAD — Isotretinoin severe acne",          "https://www.aad.org/public/diseases/acne/derm-treat/isotretinoin", "AAD"),
        ("DermNet NZ — Cystic acne",                "https://dermnetnz.org/topics/acne", "DermNet NZ"),
        ("PubMed — Isotretinoin efficacy severe acne", "https://pubmed.ncbi.nlm.nih.gov/?term=isotretinoin+severe+cystic+acne+efficacy+safety", "PubMed"),
    ],
    "Melanoma": [
        ("AAD — Melanoma overview",                 "https://www.aad.org/public/diseases/skin-cancer/types/common/melanoma", "AAD"),
        ("NIH NCI — Melanoma treatment",            "https://www.cancer.gov/types/skin/patient/melanoma-treatment-pdq", "NIH NCI"),
        ("DermNet NZ — Melanoma",                   "https://dermnetnz.org/topics/melanoma", "DermNet NZ"),
        ("PubMed — Melanoma pembrolizumab BRAF",    "https://pubmed.ncbi.nlm.nih.gov/?term=melanoma+pembrolizumab+nivolumab+BRAF+immunotherapy", "PubMed"),
    ],
    "Basal Cell Carcinoma": [
        ("AAD — Basal cell carcinoma",              "https://www.aad.org/public/diseases/skin-cancer/types/common/bcc", "AAD"),
        ("NIH NCI — Skin cancer",                   "https://www.cancer.gov/types/skin", "NIH NCI"),
        ("DermNet NZ — Basal cell carcinoma",       "https://dermnetnz.org/topics/basal-cell-carcinoma", "DermNet NZ"),
        ("PubMed — BCC Mohs surgery imiquimod",     "https://pubmed.ncbi.nlm.nih.gov/?term=basal+cell+carcinoma+Mohs+surgery+imiquimod+treatment", "PubMed"),
    ],
    "Squamous Cell Carcinoma (Skin)": [
        ("AAD — Squamous cell carcinoma",           "https://www.aad.org/public/diseases/skin-cancer/types/common/scc", "AAD"),
        ("NIH NCI — Skin SCC",                      "https://www.cancer.gov/types/skin/patient/skin-treatment-pdq", "NIH NCI"),
        ("DermNet NZ — Squamous cell carcinoma",    "https://dermnetnz.org/topics/squamous-cell-carcinoma", "DermNet NZ"),
    ],
    "Psoriasis": [
        ("AAD — Psoriasis overview",                "https://www.aad.org/public/diseases/psoriasis", "AAD"),
        ("DermNet NZ — Psoriasis",                  "https://dermnetnz.org/topics/psoriasis", "DermNet NZ"),
        ("NIH MedlinePlus — Psoriasis",             "https://medlineplus.gov/psoriasis.html", "NIH MedlinePlus"),
        ("PubMed — Psoriasis biologics adalimumab", "https://pubmed.ncbi.nlm.nih.gov/?term=psoriasis+biologic+adalimumab+secukinumab+treatment", "PubMed"),
    ],
    "Eczema / Atopic Dermatitis": [
        ("AAD — Eczema overview",                   "https://www.aad.org/public/diseases/eczema", "AAD"),
        ("DermNet NZ — Atopic dermatitis",          "https://dermnetnz.org/topics/atopic-dermatitis", "DermNet NZ"),
        ("NIH MedlinePlus — Eczema",                "https://medlineplus.gov/eczema.html", "NIH MedlinePlus"),
        ("PubMed — Dupilumab atopic dermatitis",    "https://pubmed.ncbi.nlm.nih.gov/?term=atopic+dermatitis+dupilumab+treatment+efficacy", "PubMed"),
    ],
    "Rosacea": [
        ("AAD — Rosacea overview",                  "https://www.aad.org/public/diseases/rosacea", "AAD"),
        ("DermNet NZ — Rosacea",                    "https://dermnetnz.org/topics/rosacea", "DermNet NZ"),
        ("NIH MedlinePlus — Rosacea",               "https://medlineplus.gov/rosacea.html", "NIH MedlinePlus"),
    ],
    "Vitiligo": [
        ("AAD — Vitiligo",                          "https://www.aad.org/public/diseases/a-z/vitiligo-overview", "AAD"),
        ("DermNet NZ — Vitiligo",                   "https://dermnetnz.org/topics/vitiligo", "DermNet NZ"),
        ("NIH MedlinePlus — Vitiligo",              "https://medlineplus.gov/vitiligo.html", "NIH MedlinePlus"),
    ],
    "Ringworm / Tinea Corporis": [
        ("AAD — Ringworm overview",                 "https://www.aad.org/public/diseases/a-z/ringworm-overview", "AAD"),
        ("DermNet NZ — Tinea corporis",             "https://dermnetnz.org/topics/tinea-corporis", "DermNet NZ"),
        ("CDC — Ringworm",                          "https://www.cdc.gov/fungal/diseases/ringworm/index.html", "CDC"),
    ],
    "Contact Dermatitis": [
        ("AAD — Contact dermatitis",                "https://www.aad.org/public/diseases/eczema/types/contact-dermatitis", "AAD"),
        ("DermNet NZ — Contact dermatitis",         "https://dermnetnz.org/topics/contact-dermatitis", "DermNet NZ"),
        ("NIH MedlinePlus — Contact dermatitis",    "https://medlineplus.gov/ency/article/000869.htm", "NIH MedlinePlus"),
    ],
    "Urticaria (Hives)": [
        ("AAD — Hives overview",                    "https://www.aad.org/public/diseases/a-z/hives-overview", "AAD"),
        ("DermNet NZ — Urticaria",                  "https://dermnetnz.org/topics/urticaria", "DermNet NZ"),
        ("NIH MedlinePlus — Hives",                 "https://medlineplus.gov/hives.html", "NIH MedlinePlus"),
    ],
    "Herpes Simplex / Zoster": [
        ("DermNet NZ — Herpes simplex",             "https://dermnetnz.org/topics/herpes-simplex", "DermNet NZ"),
        ("CDC — Shingles (Herpes Zoster)",          "https://www.cdc.gov/shingles/index.html", "CDC"),
        ("NIH MedlinePlus — Herpes zoster",         "https://medlineplus.gov/shingles.html", "NIH MedlinePlus"),
    ],
    "Warts (Verruca)": [
        ("AAD — Warts overview",                    "https://www.aad.org/public/diseases/a-z/warts-overview", "AAD"),
        ("DermNet NZ — Viral warts",                "https://dermnetnz.org/topics/viral-warts", "DermNet NZ"),
        ("NIH MedlinePlus — Warts",                 "https://medlineplus.gov/warts.html", "NIH MedlinePlus"),
    ],
    "Seborrheic Keratosis": [
        ("AAD — Seborrheic keratosis",              "https://www.aad.org/public/diseases/a-z/seborrheic-keratoses-overview", "AAD"),
        ("DermNet NZ — Seborrhoeic keratosis",      "https://dermnetnz.org/topics/seborrhoeic-keratosis", "DermNet NZ"),
    ],
    "Cellulitis": [
        ("AAD — Cellulitis",                        "https://www.aad.org/public/diseases/a-z/cellulitis-overview", "AAD"),
        ("DermNet NZ — Cellulitis",                 "https://dermnetnz.org/topics/cellulitis", "DermNet NZ"),
        ("NIH MedlinePlus — Cellulitis",            "https://medlineplus.gov/cellulitis.html", "NIH MedlinePlus"),
        ("PubMed — Cellulitis cephalexin vancomycin","https://pubmed.ncbi.nlm.nih.gov/?term=cellulitis+antibiotic+cephalexin+vancomycin+MRSA", "PubMed"),
    ],
    "Impetigo": [
        ("AAD — Impetigo",                          "https://www.aad.org/public/diseases/a-z/impetigo-overview", "AAD"),
        ("DermNet NZ — Impetigo",                   "https://dermnetnz.org/topics/impetigo", "DermNet NZ"),
        ("CDC — Impetigo",                          "https://www.cdc.gov/group-a-strep/hcp/clinical-guidance/impetigo.html", "CDC"),
    ],
    "Lipoma / Sebaceous Cyst": [
        ("DermNet NZ — Lipoma",                     "https://dermnetnz.org/topics/lipoma", "DermNet NZ"),
        ("AAD — Lipoma overview",                   "https://www.aad.org/public/diseases/a-z/lipomas", "AAD"),
        ("NIH MedlinePlus — Lipoma",                "https://medlineplus.gov/ency/article/003279.htm", "NIH MedlinePlus"),
    ],
    "Other Skin Condition": [
        ("DermNet NZ — Dermatology A-Z",            "https://dermnetnz.org/topics", "DermNet NZ"),
        ("AAD — Skin conditions A-Z",               "https://www.aad.org/public/diseases/a-z", "AAD"),
        ("NIH MedlinePlus — Skin conditions",       "https://medlineplus.gov/skinconditions.html", "NIH MedlinePlus"),
    ],

    # ══════════════════════════════════════════════════════════════════════════
    # X-RAY / BONE & JOINT  (14 labels)
    # ══════════════════════════════════════════════════════════════════════════

    "No Fracture Detected": [
        ("Radiopaedia — Normal bone variants",      "https://radiopaedia.org/articles/normal-skeletal-variants", "Radiopaedia"),
        ("NIH MedlinePlus — Bone health",           "https://medlineplus.gov/bonestrength.html", "NIH MedlinePlus"),
    ],
    "Simple Fracture": [
        ("Radiopaedia — Fracture types",            "https://radiopaedia.org/articles/fracture-types", "Radiopaedia"),
        ("AAOS — Fractures overview",               "https://orthoinfo.aaos.org/en/diseases--conditions/fractures-broken-bones/", "AAOS"),
        ("NIH MedlinePlus — Fractures",             "https://medlineplus.gov/fractures.html", "NIH MedlinePlus"),
    ],
    "Comminuted Fracture": [
        ("Radiopaedia — Comminuted fracture",       "https://radiopaedia.org/articles/comminuted-fracture", "Radiopaedia"),
        ("AAOS — Complex fractures surgical repair","https://orthoinfo.aaos.org/en/diseases--conditions/fractures-broken-bones/", "AAOS"),
        ("PubMed — Comminuted fracture ORIF",       "https://pubmed.ncbi.nlm.nih.gov/?term=comminuted+fracture+ORIF+surgical+fixation+outcomes", "PubMed"),
    ],
    "Stress Fracture / Hairline Fracture": [
        ("Radiopaedia — Stress fracture",           "https://radiopaedia.org/articles/stress-fracture", "Radiopaedia"),
        ("AAOS — Stress fractures",                 "https://orthoinfo.aaos.org/en/diseases--conditions/stress-fractures/", "AAOS"),
        ("NIH MedlinePlus — Stress fractures",      "https://medlineplus.gov/ency/article/001205.htm", "NIH MedlinePlus"),
    ],
    "Avulsion Fracture": [
        ("Radiopaedia — Avulsion fracture",         "https://radiopaedia.org/articles/avulsion-fracture", "Radiopaedia"),
        ("AAOS — Avulsion fractures",               "https://orthoinfo.aaos.org/en/diseases--conditions/fractures-broken-bones/", "AAOS"),
    ],
    "Dislocation": [
        ("Radiopaedia — Joint dislocation",         "https://radiopaedia.org/articles/dislocation", "Radiopaedia"),
        ("AAOS — Dislocations",                     "https://orthoinfo.aaos.org/en/diseases--conditions/dislocations/", "AAOS"),
        ("NIH MedlinePlus — Dislocations",          "https://medlineplus.gov/dislocations.html", "NIH MedlinePlus"),
    ],
    "Fracture-Dislocation": [
        ("Radiopaedia — Fracture dislocation",      "https://radiopaedia.org/articles/fracture-dislocation", "Radiopaedia"),
        ("AAOS — Fracture-dislocations",            "https://orthoinfo.aaos.org/en/diseases--conditions/fractures-broken-bones/", "AAOS"),
        ("PubMed — Fracture dislocation surgical emergency", "https://pubmed.ncbi.nlm.nih.gov/?term=fracture+dislocation+ORIF+vascular+injury+emergency", "PubMed"),
    ],
    "Growth Plate Fracture (Salter-Harris)": [
        ("Radiopaedia — Salter-Harris fractures",   "https://radiopaedia.org/articles/salter-harris-fractures", "Radiopaedia"),
        ("AAOS — Growth plate fractures",           "https://orthoinfo.aaos.org/en/diseases--conditions/growth-plate-fractures/", "AAOS"),
        ("NIH MedlinePlus — Growth plate fractures","https://medlineplus.gov/ency/article/001211.htm", "NIH MedlinePlus"),
    ],
    "Bone Tumor / Lesion": [
        ("Radiopaedia — Bone tumors",               "https://radiopaedia.org/articles/bone-tumours", "Radiopaedia"),
        ("NIH NCI — Bone cancer",                   "https://www.cancer.gov/types/bone", "NIH NCI"),
        ("AAOS — Bone tumors",                      "https://orthoinfo.aaos.org/en/diseases--conditions/bone-tumor/", "AAOS"),
        ("PubMed — Osteosarcoma treatment",         "https://pubmed.ncbi.nlm.nih.gov/?term=bone+tumor+osteosarcoma+doxorubicin+cisplatin+surgery", "PubMed"),
    ],
    "Osteomyelitis (Bone Infection)": [
        ("Radiopaedia — Osteomyelitis",             "https://radiopaedia.org/articles/osteomyelitis", "Radiopaedia"),
        ("NIH MedlinePlus — Osteomyelitis",         "https://medlineplus.gov/osteomyelitis.html", "NIH MedlinePlus"),
        ("PubMed — Osteomyelitis IV antibiotics",   "https://pubmed.ncbi.nlm.nih.gov/?term=osteomyelitis+IV+antibiotics+vancomycin+cefazolin+treatment", "PubMed"),
    ],
    "Osteoporosis / Osteopenia": [
        ("Radiopaedia — Osteoporosis",              "https://radiopaedia.org/articles/osteoporosis", "Radiopaedia"),
        ("NIH — Osteoporosis",                      "https://www.bones.nih.gov/health-info/bone/osteoporosis/overview", "NIH"),
        ("Mayo Clinic — Osteoporosis",              "https://www.mayoclinic.org/diseases-conditions/osteoporosis/symptoms-causes/syc-20351968", "Mayo Clinic"),
    ],
    "Joint Space Narrowing (Arthritis)": [
        ("Radiopaedia — Osteoarthritis",            "https://radiopaedia.org/articles/osteoarthritis", "Radiopaedia"),
        ("AAOS — Arthritis overview",               "https://orthoinfo.aaos.org/en/diseases--conditions/arthritis-of-the-hip/", "AAOS"),
        ("NIH MedlinePlus — Osteoarthritis",        "https://medlineplus.gov/osteoarthritis.html", "NIH MedlinePlus"),
    ],
    "Soft Tissue Injury Only": [
        ("Radiopaedia — Soft tissue injuries",      "https://radiopaedia.org/articles/soft-tissue-injuries", "Radiopaedia"),
        ("AAOS — Sprains and strains",              "https://orthoinfo.aaos.org/en/diseases--conditions/sprains-strains-and-other-soft-tissue-injuries/", "AAOS"),
        ("NIH MedlinePlus — Sprains and strains",   "https://medlineplus.gov/sprainsandstrains.html", "NIH MedlinePlus"),
    ],
    "Other Musculoskeletal Finding": [
        ("Radiopaedia — Musculoskeletal",           "https://radiopaedia.org/articles/musculoskeletal", "Radiopaedia"),
        ("AAOS — Conditions A-Z",                   "https://orthoinfo.aaos.org/en/diseases--conditions/", "AAOS"),
        ("NIH MedlinePlus — Musculoskeletal",       "https://medlineplus.gov/musculoskeletaldisorders.html", "NIH MedlinePlus"),
    ],

    # ══════════════════════════════════════════════════════════════════════════
    # GENERAL SCAN  (13 labels — broad fallbacks)
    # ══════════════════════════════════════════════════════════════════════════

    "Malignant Mass / Cancer": [
        ("NIH NCI — Cancer types",                  "https://www.cancer.gov/types", "NIH NCI"),
        ("Radiopaedia — Malignant masses",          "https://radiopaedia.org/articles/malignant-tumours", "Radiopaedia"),
        ("Mayo Clinic — Cancer overview",           "https://www.mayoclinic.org/diseases-conditions/cancer/symptoms-causes/syc-20370588", "Mayo Clinic"),
    ],
    "Suspicious Lesion": [
        ("Radiopaedia — Lesion characterisation",   "https://radiopaedia.org/articles/lesion", "Radiopaedia"),
        ("NIH NCI — When cancer is found",          "https://www.cancer.gov/types/common-cancers", "NIH NCI"),
        ("Mayo Clinic — Understanding suspicious findings", "https://www.mayoclinic.org/tests-procedures/biopsy/about/pac-20384604", "Mayo Clinic"),
    ],
    "Infection / Abscess": [
        ("Radiopaedia — Abscess imaging",           "https://radiopaedia.org/articles/abscess", "Radiopaedia"),
        ("NIH MedlinePlus — Abscess",               "https://medlineplus.gov/abscess.html", "NIH MedlinePlus"),
        ("CDC — Bacterial infections",              "https://www.cdc.gov/niosh/topics/emres/biologic.html", "CDC"),
    ],
    "Inflammatory Condition": [
        ("NIH MedlinePlus — Inflammation",          "https://medlineplus.gov/chronicinflammation.html", "NIH MedlinePlus"),
        ("Radiopaedia — Inflammatory lesions",      "https://radiopaedia.org/articles/inflammatory-lesions", "Radiopaedia"),
    ],
    "Fracture / Trauma": [
        ("Radiopaedia — Fracture types",            "https://radiopaedia.org/articles/fracture-types", "Radiopaedia"),
        ("AAOS — Fractures",                        "https://orthoinfo.aaos.org/en/diseases--conditions/fractures-broken-bones/", "AAOS"),
        ("NIH MedlinePlus — Fractures",             "https://medlineplus.gov/fractures.html", "NIH MedlinePlus"),
    ],
    "Skin Condition": [
        ("DermNet NZ — Dermatology A-Z",            "https://dermnetnz.org/topics", "DermNet NZ"),
        ("AAD — Skin conditions A-Z",               "https://www.aad.org/public/diseases/a-z", "AAD"),
        ("NIH MedlinePlus — Skin conditions",       "https://medlineplus.gov/skinconditions.html", "NIH MedlinePlus"),
    ],
    "Neurological Finding": [
        ("Radiopaedia — Neuroradiology",            "https://radiopaedia.org/articles/neuroradiology", "Radiopaedia"),
        ("NIH NINDS — Neurological disorders",      "https://www.ninds.nih.gov/health-information/disorders", "NIH NINDS"),
        ("Mayo Clinic — Neurological symptoms",     "https://www.mayoclinic.org/symptoms/neurological-symptoms/basics/definition/sym-20050938", "Mayo Clinic"),
    ],
    "Cardiovascular Finding": [
        ("Radiopaedia — Cardiac imaging",           "https://radiopaedia.org/articles/cardiac-radiology", "Radiopaedia"),
        ("AHA — Heart conditions",                  "https://www.heart.org/en/health-topics", "AHA"),
        ("NIH NHLBI — Heart and vascular",          "https://www.nhlbi.nih.gov/health-topics", "NIH NHLBI"),
    ],
    "Respiratory Finding": [
        ("Radiopaedia — Chest radiology",           "https://radiopaedia.org/articles/chest-radiology", "Radiopaedia"),
        ("NIH NHLBI — Lung diseases",               "https://www.nhlbi.nih.gov/health-topics/all-topics#c", "NIH NHLBI"),
        ("Mayo Clinic — Lung conditions",           "https://www.mayoclinic.org/diseases-conditions", "Mayo Clinic"),
    ],
    "Gastrointestinal Finding": [
        ("Radiopaedia — Gastrointestinal radiology","https://radiopaedia.org/articles/gastrointestinal-radiology", "Radiopaedia"),
        ("NIH MedlinePlus — Digestive diseases",    "https://medlineplus.gov/digestivediseases.html", "NIH MedlinePlus"),
        ("Mayo Clinic — Digestive health",          "https://www.mayoclinic.org/diseases-conditions/digestive-diseases", "Mayo Clinic"),
    ],
    "Musculoskeletal Finding": [
        ("Radiopaedia — Musculoskeletal",           "https://radiopaedia.org/articles/musculoskeletal", "Radiopaedia"),
        ("AAOS — Conditions A-Z",                   "https://orthoinfo.aaos.org/en/diseases--conditions/", "AAOS"),
        ("NIH MedlinePlus — Bone joints muscles",   "https://medlineplus.gov/bonesdiseasesconditions.html", "NIH MedlinePlus"),
    ],

    # ══════════════════════════════════════════════════════════════════════════
    # SHARED NORMALS & FALLBACKS
    # ══════════════════════════════════════════════════════════════════════════

    "No Abnormality Detected": [
        ("NIH MedlinePlus — Medical imaging",       "https://medlineplus.gov/medicalimaging.html", "NIH MedlinePlus"),
        ("Radiopaedia — Normal variants",           "https://radiopaedia.org/articles/normal-variants", "Radiopaedia"),
    ],
    "Other Abnormality": [
        ("Radiopaedia — Image library",             "https://radiopaedia.org/", "Radiopaedia"),
        ("NIH MedlinePlus — Medical encyclopedia",  "https://medlineplus.gov/encyclopedia.html", "NIH MedlinePlus"),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_references(label: str, max_refs: int = 4) -> list[dict]:
    """
    Return up to max_refs reference dicts for the given label.

    Each dict:
        title  : str  — display text
        url    : str  — full URL
        source : str  — short source name
        color  : str  — hex colour for the source badge
    """
    entries = REFERENCE_MAP.get(label)

    # Partial match — e.g. "Alzheimer's Disease" matches any severity variant
    if not entries:
        label_base = label.lower().split(" (")[0]
        for key, val in REFERENCE_MAP.items():
            if label_base == key.lower().split(" (")[0]:
                entries = val
                break

    # Final fallback
    if not entries:
        entries = REFERENCE_MAP.get("Other Abnormality", [])

    return [
        {
            "title":  title,
            "url":    url,
            "source": source,
            "color":  SOURCE_COLORS.get(source, "#64748b"),
        }
        for title, url, source in entries[:max_refs]
    ]


def get_source_label(label: str) -> str:
    """Return the primary source name for a label (used in meta table)."""
    refs = get_references(label, max_refs=1)
    return refs[0]["source"] if refs else "Radiopaedia"