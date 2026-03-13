"""
TCIA (The Cancer Imaging Archive) Integration Service
======================================================

What TCIA does for your system:
  After Claude Vision detects a cancer type, this service queries the TCIA
  public API to find REAL, clinically-verified cancer cases that match the
  detected diagnosis. These are shown on the result page as reference cases,
  giving clinicians context from thousands of real patient studies.

TCIA API base: https://services.cancerimagingarchive.net/nbia-api/services/v1/
No API key required for public collections.

Collections used:
  - TCGA-GBM       → Glioblastoma (brain)
  - TCGA-LGG       → Low-grade glioma / Astrocytoma
  - TCGA-BRCA      → Breast cancer
  - TCGA-LUAD      → Lung Adenocarcinoma
  - TCGA-LUSC      → Squamous Cell Lung Cancer
  - TCGA-LIHC      → Hepatocellular Carcinoma
  - TCGA-KIRC      → Renal Cell Carcinoma
  - TCGA-PAAD      → Pancreatic Adenocarcinoma
  - CBIS-DDSM      → Breast mammography (mass/calcification)
  - LIDC-IDRI      → Lung nodules (CT)
  - Meningioma-SEG → Meningioma MRI segmentations
"""

import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)

TCIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
TCIA_TIMEOUT = 10  # seconds

# Shared session with proper headers
_session = requests.Session()
_session.headers.update({
    "Accept": "application/json",
    "Content-Type": "application/json",
})

# ── Cancer type → TCIA collection mapping ─────────────────────────────────────
CANCER_TO_COLLECTION = {
    # Brain
    "Glioblastoma (GBM)":               {"collection": "TCGA-GBM",      "modality": "MR",  "body_part": "BRAIN"},
    "Astrocytoma (Low/High Grade)":      {"collection": "TCGA-LGG",      "modality": "MR",  "body_part": "BRAIN"},
    "Malignant Glioma":                  {"collection": "TCGA-GBM",      "modality": "MR",  "body_part": "BRAIN"},
    "Glioma / Brain Tumor":              {"collection": "TCGA-LGG",      "modality": "MR",  "body_part": "BRAIN"},
    "Meningioma":                        {"collection": "Meningioma-SEG", "modality": "MR",  "body_part": "BRAIN"},
    "Brain Metastasis":                  {"collection": "TCGA-GBM",      "modality": "MR",  "body_part": "BRAIN"},
    "Primary CNS Lymphoma":              {"collection": "TCGA-GBM",      "modality": "MR",  "body_part": "BRAIN"},
    "Glioblastoma / Brain Tumor":        {"collection": "TCGA-GBM",      "modality": "MR",  "body_part": "BRAIN"},

    # Lung / Chest
    "Lung Adenocarcinoma":               {"collection": "TCGA-LUAD",     "modality": "CT",  "body_part": "CHEST"},
    "Squamous Cell Carcinoma":           {"collection": "TCGA-LUSC",     "modality": "CT",  "body_part": "CHEST"},
    "Small Cell Lung Cancer":            {"collection": "TCGA-LUSC",     "modality": "CT",  "body_part": "CHEST"},
    "Lung Cancer":                       {"collection": "LIDC-IDRI",     "modality": "CT",  "body_part": "CHEST"},
    "Pulmonary Metastasis":              {"collection": "LIDC-IDRI",     "modality": "CT",  "body_part": "CHEST"},

    # Abdominal
    "Hepatocellular Carcinoma (HCC)":    {"collection": "TCGA-LIHC",     "modality": "MR",  "body_part": "LIVER"},
    "Renal Cell Carcinoma":              {"collection": "TCGA-KIRC",     "modality": "CT",  "body_part": "KIDNEY"},
    "Pancreatic Adenocarcinoma":         {"collection": "TCGA-PAAD",     "modality": "CT",  "body_part": "PANCREAS"},
    "Pancreatic Adenocarcinoma (PDAC)":  {"collection": "TCGA-PAAD",     "modality": "CT",  "body_part": "PANCREAS"},

    # Breast
    "Breast Cancer":                     {"collection": "TCGA-BRCA",     "modality": "MR",  "body_part": "BREAST"},
}

# Generic fallback by modality/body part
MODALITY_FALLBACK = {
    "brain_mri":   {"collection": "TCGA-GBM",   "modality": "MR", "body_part": "BRAIN"},
    "chest_xray":  {"collection": "LIDC-IDRI",  "modality": "CT", "body_part": "CHEST"},
    "ct_scan":     {"collection": "TCGA-GBM",   "modality": "CT", "body_part": "BRAIN"},
    "abdominal":   {"collection": "TCGA-LIHC",  "modality": "CT", "body_part": "LIVER"},
    "mri_spine":   {"collection": "TCGA-LGG",   "modality": "MR", "body_part": "SPINE"},
    "general":     {"collection": "TCGA-GBM",   "modality": "MR", "body_part": "BRAIN"},
}

# Ordered fallback chain — if the primary collection returns empty, try these
COLLECTION_FALLBACK_CHAIN = [
    {"collection": "TCGA-GBM",  "modality": "MR", "body_part": "BRAIN"},
    {"collection": "LIDC-IDRI", "modality": "CT", "body_part": "CHEST"},
    {"collection": "TCGA-BRCA", "modality": "MR", "body_part": "BREAST"},
]


def _safe_json(resp: requests.Response) -> Optional[list]:
    """
    Safely parse a TCIA API response as JSON.
    Returns None if the body is empty or not valid JSON.
    """
    raw = resp.text.strip()
    if not raw:
        return None
    try:
        return resp.json()
    except ValueError:
        logger.warning(f"TCIA non-JSON response ({resp.status_code}): {raw[:200]}")
        return None


def _fetch_series(collection: str, modality: str) -> Optional[list]:
    """
    Call getSeries for a given collection + modality.
    Returns a list of series dicts, or None on any failure.
    """
    try:
        resp = _session.get(
            f"{TCIA_BASE}/getSeries",
            params={
                "Collection": collection,
                "Modality":   modality,
            },
            timeout=TCIA_TIMEOUT,
        )
        if not resp.ok:
            logger.warning(f"TCIA getSeries HTTP {resp.status_code} for {collection}/{modality}")
            return None
        data = _safe_json(resp)
        if data is None:
            logger.warning(f"TCIA getSeries empty body for {collection}/{modality}")
        return data
    except requests.exceptions.Timeout:
        logger.warning(f"TCIA getSeries timed out for {collection}/{modality}")
        return None
    except Exception as exc:
        logger.error(f"TCIA getSeries error for {collection}/{modality}: {exc}")
        return None


def _fetch_series_size(series_uid: str) -> dict:
    """
    Fetch image count / byte size for one series.
    Returns a dict with ObjectCount and TotalSizeInBytes, or empty dict on failure.
    """
    if not series_uid:
        return {}
    try:
        resp = _session.get(
            f"{TCIA_BASE}/getSeriesSize",
            params={"SeriesInstanceUID": series_uid},
            timeout=TCIA_TIMEOUT,
        )
        if not resp.ok:
            return {}
        data = _safe_json(resp)
        if data and isinstance(data, list) and len(data) > 0:
            return {
                "ObjectCount":      data[0].get("ObjectCount", "N/A"),
                "TotalSizeInBytes": data[0].get("TotalSizeInBytes", "N/A"),
            }
    except Exception:
        pass
    return {}


def _clean_label(label: str) -> str:
    """Remove simulation suffix and parentheticals for matching."""
    return label.replace(" (SIMULATION)", "").strip()


def get_tcia_reference_cases(
    cancer_label: str,
    scan_type: str = "general",
    max_results: int = 3,
) -> dict:
    """
    Query TCIA for real verified cancer cases matching the detected cancer type.

    Args:
        cancer_label: The result_label from the AI prediction (e.g. "Glioblastoma (GBM)")
        scan_type:    The upload form scan_type (e.g. "brain_mri")
        max_results:  How many reference series to return

    Returns:
        dict with keys:
          - collection: TCIA collection name used
          - series: list of series metadata dicts
          - patient_count: number of matching patients found
          - total_series: total series count in this collection
          - tcia_url: link to browse collection on TCIA website
          - modality / body_part: echo of what was queried
          - error: error message string if failed, else None
    """
    clean = _clean_label(cancer_label)

    # Pick the best matching TCIA collection config
    config = (
        CANCER_TO_COLLECTION.get(clean)
        or MODALITY_FALLBACK.get(scan_type)
        or MODALITY_FALLBACK["general"]
    )

    collection = config["collection"]
    modality   = config["modality"]
    body_part  = config["body_part"]
    tcia_url   = f"https://www.cancerimagingarchive.net/collection/{collection.lower()}/"

    # ── Try the primary collection, then fall back if empty ──────────────────
    all_series = _fetch_series(collection, modality)

    if not all_series:
        logger.info(f"Primary collection {collection} empty, trying fallbacks…")
        for fallback in COLLECTION_FALLBACK_CHAIN:
            if fallback["collection"] == collection:
                continue  # already tried
            all_series = _fetch_series(fallback["collection"], fallback["modality"])
            if all_series:
                collection = fallback["collection"]
                modality   = fallback["modality"]
                body_part  = fallback["body_part"]
                tcia_url   = f"https://www.cancerimagingarchive.net/collection/{collection.lower()}/"
                logger.info(f"Using fallback collection: {collection}")
                break

    if not all_series:
        return {
            "collection":    collection,
            "series":        [],
            "patient_count": 0,
            "total_series":  0,
            "tcia_url":      tcia_url,
            "modality":      modality,
            "body_part":     body_part,
            "error":         f"No series data available for {collection} (modality={modality}). TCIA may be temporarily unavailable.",
        }

    # ── Enrich a sample of series with per-series metadata ───────────────────
    sampled  = all_series[:max_results]
    enriched = []

    for s in sampled:
        series_uid = s.get("SeriesInstanceUID", "")
        entry = {
            "SeriesInstanceUID": series_uid,
            "PatientID":         s.get("PatientID", "Anonymous"),
            "StudyDate":         s.get("StudyDate", "Unknown"),
            "Modality":          s.get("Modality", modality),
            "BodyPartExamined":  s.get("BodyPartExamined", body_part),
            "SeriesDescription": s.get("SeriesDescription", ""),
            "Manufacturer":      s.get("Manufacturer", ""),
            "Collection":        collection,
            "tcia_url":          tcia_url,
        }

        # Fetch image count (best-effort, don't fail the whole request)
        size_info = _fetch_series_size(series_uid)
        entry.update(size_info)

        enriched.append(entry)

    # Count unique patients across all series
    patient_ids = {s.get("PatientID", "") for s in all_series if s.get("PatientID")}

    return {
        "collection":    collection,
        "series":        enriched,
        "patient_count": len(patient_ids),
        "total_series":  len(all_series),
        "tcia_url":      tcia_url,
        "body_part":     body_part,
        "modality":      modality,
        "error":         None,
    }


def get_collection_stats(collection: str) -> dict:
    """
    Get high-level stats for a TCIA collection.
    Returns patient count, modalities, body parts, or empty dict on failure.
    """
    try:
        resp = _session.get(
            f"{TCIA_BASE}/getCollectionValues",
            timeout=TCIA_TIMEOUT,
        )
        resp.raise_for_status()
        data = _safe_json(resp)
        if not data:
            return {}
        match = next((c for c in data if c.get("Collection") == collection), None)
        return match or {}
    except Exception as exc:
        logger.error(f"TCIA getCollectionValues error: {exc}")
        return {}


def is_tcia_available() -> bool:
    """Quick ping to check if TCIA API is reachable."""
    try:
        resp = _session.get(
            f"{TCIA_BASE}/getCollectionValues",
            timeout=4,
        )
        return resp.ok and _safe_json(resp) is not None
    except Exception:
        return False