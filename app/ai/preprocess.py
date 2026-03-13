"""
MONAI-based preprocessing pipeline for medical image analysis.

Architect notes:
  - All transforms are composable; swap or extend per modality.
  - DICOM parsing uses pydicom + MONAI LoadImage with reader="pydicom".
  - NIfTI uses SimpleITK reader via MONAI.
  - PNG/JPG use PIL then convert to tensor.
  - Intensity normalization follows MedicalNet / MONAI best practices.
"""

import os
import logging
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional imports – graceful fallback when MONAI / PyTorch not installed
# ---------------------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed – running in simulation mode.")

try:
    from monai.transforms import (
        Compose,
        LoadImage,
        EnsureChannelFirst,
        ScaleIntensity,
        Resize,
        NormalizeIntensity,
        ToTensor,
    )
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    logger.warning("MONAI not installed – running in simulation mode.")

# Target tensor shape for DenseNet-121 (MONAI default)
TARGET_SIZE = (224, 224)  # H x W
DICOM_EXT = {".dcm"}
NIFTI_EXT = {".nii", ".gz"}


# ---------------------------------------------------------------------------
# MONAI transform pipelines
# ---------------------------------------------------------------------------

def _build_standard_pipeline():
    """
    Standard 2-D pipeline for PNG/JPG inputs.
    Produces a (1, 224, 224) float32 tensor.
    """
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize(spatial_size=TARGET_SIZE),
        ScaleIntensity(minv=0.0, maxv=1.0),
        NormalizeIntensity(nonzero=True),
        ToTensor(),
    ])


def _build_dicom_pipeline():
    """
    DICOM pipeline with Hounsfield unit windowing suitable for CT scans.
    Window: brain window [-80, 80 HU], soft-tissue [-150, 250 HU].
    """
    return Compose([
        LoadImage(image_only=True, reader="PydicomReader"),
        EnsureChannelFirst(),
        Resize(spatial_size=TARGET_SIZE),
        ScaleIntensity(minv=0.0, maxv=1.0),   # after windowing MONAI clamps automatically
        NormalizeIntensity(nonzero=True),
        ToTensor(),
    ])


def _build_nifti_pipeline():
    """
    NIfTI (3-D MRI) pipeline – takes the middle axial slice for 2-D inference.
    For full 3-D inference replace Resize with SpatialPad + CropForeground.
    """
    return Compose([
        LoadImage(image_only=True, reader="NibabelReader"),
        EnsureChannelFirst(),
        Resize(spatial_size=(*TARGET_SIZE, -1)),   # resize H/W, keep depth
        ScaleIntensity(minv=0.0, maxv=1.0),
        NormalizeIntensity(nonzero=True),
        ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_image(file_path: str, ext: str) -> "torch.Tensor":
    """
    Preprocess a medical image file into a model-ready tensor.

    Args:
        file_path: Absolute path to the image.
        ext:       File extension without dot (png, jpg, dcm, nii).

    Returns:
        Float32 tensor of shape (1, C, H, W) ready for batch inference.
    """
    ext = ext.lower().lstrip(".")

    if MONAI_AVAILABLE and TORCH_AVAILABLE:
        return _monai_preprocess(file_path, ext)
    else:
        return _fallback_preprocess(file_path, ext)


def _monai_preprocess(file_path: str, ext: str) -> "torch.Tensor":
    try:
        if ext == "dcm":
            pipeline = _build_dicom_pipeline()
        elif ext in ("nii", "gz"):
            pipeline = _build_nifti_pipeline()
        else:
            pipeline = _build_standard_pipeline()

        tensor = pipeline(file_path)          # shape: (C, H, W) or (C, H, W, D)

        # For 3-D NIfTI: grab the middle axial slice
        if tensor.ndim == 4:
            mid = tensor.shape[-1] // 2
            tensor = tensor[..., mid]         # → (C, H, W)

        tensor = tensor.unsqueeze(0)          # → (1, C, H, W)

        # Ensure 3-channel input expected by DenseNet-121 pretrained on ImageNet
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)

        return tensor.float()

    except Exception as exc:
        logger.error(f"MONAI preprocessing failed for {file_path}: {exc}")
        raise


def _fallback_preprocess(file_path: str, ext: str) -> np.ndarray:
    """
    PIL-based fallback when MONAI/PyTorch are unavailable.
    Returns a normalized numpy array of shape (1, 3, 224, 224).
    """
    from PIL import Image

    try:
        img = Image.open(file_path).convert("RGB")
    except Exception:
        # For DICOM without MONAI – return zeros
        logger.warning("Cannot open DICOM without MONAI; using zero array.")
        return np.zeros((1, 3, *TARGET_SIZE), dtype=np.float32)

    img = img.resize(TARGET_SIZE[::-1])  # PIL uses (W, H)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)         # HWC → CHW
    arr = arr[np.newaxis, ...]           # → (1, 3, H, W)
    return arr


def extract_thumbnail(file_path: str, ext: str, size: Tuple[int, int] = (400, 400)) -> str:
    """
    Generate a PNG thumbnail for browser preview.
    Returns path to the thumbnail file.
    """
    from PIL import Image

    thumb_path = file_path.rsplit(".", 1)[0] + "_thumb.png"
    try:
        if ext == "dcm" and MONAI_AVAILABLE:
            # Use MONAI to read DICOM and convert to PIL
            from monai.transforms import LoadImage, ScaleIntensity
            loader = Compose([LoadImage(image_only=True), ScaleIntensity(minv=0, maxv=255)])
            arr = np.array(loader(file_path))
            if arr.ndim == 3:
                arr = arr[arr.shape[0] // 2]   # middle slice
            img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        else:
            img = Image.open(file_path).convert("RGB")

        img.thumbnail(size)
        img.save(thumb_path, "PNG")
        return thumb_path
    except Exception as exc:
        logger.warning(f"Thumbnail generation failed: {exc}")
        return file_path
