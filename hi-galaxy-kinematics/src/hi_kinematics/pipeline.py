from pathlib import Path

import numpy as np
from astropy.io import fits

from hi_kinematics.cube import load_fits_cube
from hi_kinematics.denoise import wavelet_denoise_3d


# ============================================================
# Global paths
# ============================================================

RAW_CUBE_PATH = Path(
    r"C:\Users\ASUS\Desktop\NGC_2403cube.fits"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GALAXY_ID = "_".join(RAW_CUBE_PATH.stem.split("_")[:2])

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / GALAXY_ID

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DENOISED_CUBE_PATH = (
    PROCESSED_DIR / f"{RAW_CUBE_PATH.stem}_denoised.fits"
)

# This is the path all downstream code should use.
ACTIVE_CUBE_PATH = RAW_CUBE_PATH


# ============================================================
# Denoising pipeline
# ============================================================

def prepare_cube():
    """
    Denoise the raw cube once and update ACTIVE_CUBE_PATH
    to point to the denoised FITS file.
    """
    global ACTIVE_CUBE_PATH

    if not RAW_CUBE_PATH.exists():
        raise FileNotFoundError(
            f"Raw FITS cube not found: {RAW_CUBE_PATH}"
        )

    if not DENOISED_CUBE_PATH.exists():
        print(f"Loading raw cube: {RAW_CUBE_PATH}")

        cube, header = load_fits_cube(RAW_CUBE_PATH)

        print("Running 3D wavelet denoising...")

        denoised_cube, signal_mask = wavelet_denoise_3d(
            cube,
            wavelet="db1",
            level=5,
            threshold_factor=2.5,
            threshold_sigma=4.0,
            grow_sigma=2.0,
            spatial_grow=2,
        )

        print(f"Signal fraction: {signal_mask.mean():.3%}")

        # Preserve the original FITS header and save the denoised cube.
        # The original raw file is not modified.
        with fits.open(RAW_CUBE_PATH, memmap=True) as hdul:
            primary_header = hdul[0].header.copy()

        primary_header["HISTORY"] = (
            "3D wavelet denoising applied; original cube preserved."
        )

        fits.PrimaryHDU(
            data=denoised_cube.astype(np.float32),
            header=primary_header,
        ).writeto(DENOISED_CUBE_PATH, overwrite=True)

        print(f"Denoised cube saved: {DENOISED_CUBE_PATH}")

    else:
        print(f"Using existing denoised cube: {DENOISED_CUBE_PATH}")

    ACTIVE_CUBE_PATH = DENOISED_CUBE_PATH

    print(f"Active cube path: {ACTIVE_CUBE_PATH}")

    return ACTIVE_CUBE_PATH