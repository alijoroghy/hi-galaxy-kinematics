"""FITS cube loading and spectral moment calculations.

Assumption: the input cube is ordered (spectral, y, x), as in the THINGS
FITS cube used in the original notebook. Verify this with FITS WCS metadata
before analysing a different data product.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


def load_fits_cube(path: str | Path, hdu_index: int = 0):
    """Return (float cube, copied FITS header) from a FITS file."""
    from astropy.io import fits
    with fits.open(path, memmap=True) as hdul:
        header = hdul[hdu_index].header.copy()
        cube = np.asarray(hdul[hdu_index].data, dtype=float)
    cube = np.squeeze(cube)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3-D cube after squeeze; got shape {cube.shape}")
    return cube, header


def spectral_axis_kms(header, n_channels: int) -> np.ndarray:
    """Compute spectral coordinates in km/s from the FITS WCS.

    Supports a spectral axis described in velocity units. For frequency axes,
    a rest frequency and an explicit Doppler convention are required; this
    function intentionally raises rather than silently mislabelling frequency
    as velocity.
    """
    from astropy.wcs import WCS
    wcs = WCS(header)
    spec_axis = None
    for i, ctype in enumerate(wcs.wcs.ctype):
        c = (ctype or "").upper()
        if any(token in c for token in ("VRAD", "VOPT", "VELO", "FELO")):
            spec_axis = i
            break
    if spec_axis is None:
        raise ValueError(
            "No velocity spectral axis found in FITS WCS. Frequency-to-velocity "
            "conversion requires rest-frequency and Doppler-convention metadata."
        )

    # FITS WCS pixel axes are 1-indexed; numpy cube axis order is reversed.
    pixel = np.arange(n_channels, dtype=float)
    pix = np.ones((n_channels, wcs.pixel_n_dim), dtype=float)
    pix[:, spec_axis] = pixel
    world = wcs.all_pix2world(pix, 0)
    values = world[:, spec_axis]
    unit_obj = wcs.wcs.cunit[spec_axis]

    if not unit_obj:
        unit_str = "m/s"
    else:
        unit_str = str(unit_obj)

        unit = unit_str.lower().replace(" ", "")

    if unit in {"m/s", "ms-1", "m.s-1", "m.s**-1"}:
        values = values / 1000.0

    elif unit in {"km/s", "kms-1", "km.s-1", "km.s**-1"}:
        pass

    else:
        raise ValueError(
        f"Unsupported velocity unit {unit!r}; "
        "convert explicitly."
    )
    return np.asarray(values, dtype=float)


def moment_maps(cube: np.ndarray, velocity_kms: np.ndarray, mask: np.ndarray | None = None):
    """Calculate integrated intensity (moment 0) and intensity-weighted velocity.

    Moment 0 includes the channel width (km/s). Moment 1 is in km/s. Non-finite
    and non-positive intensities are excluded from the weighted moments.
    """
    cube = np.asarray(cube, dtype=float)
    velocity = np.asarray(velocity_kms, dtype=float)
    if cube.ndim != 3 or velocity.ndim != 1 or len(velocity) != cube.shape[0]:
        raise ValueError("Expected cube (nchan, ny, nx) and velocity axis (nchan,).")
    if len(velocity) < 2:
        raise ValueError("At least two spectral channels are needed to estimate channel width.")
    dv = float(np.nanmedian(np.abs(np.diff(velocity))))
    valid = np.isfinite(cube) & (cube > 0)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != cube.shape:
            raise ValueError("mask must have the same shape as cube")
        valid &= mask
    intensity = np.where(valid, cube, 0.0)
    denom = intensity.sum(axis=0)
    m0 = denom * dv
    weighted = np.sum(intensity * velocity[:, None, None], axis=0)
    m1 = np.full(denom.shape, np.nan, dtype=float)
    good = denom > 0
    m1[good] = weighted[good] / denom[good]
    m0[~np.isfinite(m0)] = np.nan
    return m0, m1
