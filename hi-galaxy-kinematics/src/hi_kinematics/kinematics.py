"""Transparent, geometry-fixed tilted-ring rotation-curve fitting.

This is a first-pass kinematic model, not a replacement for TiRiFiC. Geometry
(center, position angle, inclination) is supplied by the user and held fixed;
only rotation speed and systemic velocity are fitted in each annulus.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares


def sky_to_disk(x, y, x0, y0, pa_deg, inclination_deg):
    """Return disk-plane coordinates (major, deprojected minor) in pixels.

    PA convention here is measured counter-clockwise from +x. Confirm and adapt
    this convention when comparing with a survey's astronomical PA convention.
    """
    pa = np.deg2rad(pa_deg)
    inc = np.deg2rad(inclination_deg)
    dx, dy = np.asarray(x) - x0, np.asarray(y) - y0
    major = dx * np.cos(pa) + dy * np.sin(pa)
    minor_sky = -dx * np.sin(pa) + dy * np.cos(pa)
    if abs(np.cos(inc)) < 1e-3:
        raise ValueError("Inclination too close to 90 degrees for stable deprojection.")
    minor_disk = minor_sky / np.cos(inc)
    radius = np.hypot(major, minor_disk)
    cos_theta = np.divide(major, radius, out=np.zeros_like(radius, dtype=float), where=radius > 0)
    return major, minor_disk, radius, cos_theta


def rotation_model(x, y, x0, y0, pa_deg, inclination_deg, vrot_kms, vsys_kms):
    """Projected circular-rotation velocity field in km/s."""
    _, _, _, cos_theta = sky_to_disk(x, y, x0, y0, pa_deg, inclination_deg)
    return vsys_kms + vrot_kms * np.sin(np.deg2rad(inclination_deg)) * cos_theta


def fit_rotation_curve(velocity_map, center_xy, pa_deg, inclination_deg,
                       ring_edges, valid_mask=None, min_pixels=20):
    """Fit Vrot and Vsys independently in radial annuli.

    Parameters
    ----------
    velocity_map : 2-D array
        Intensity-weighted velocity map in km/s (e.g. moment 1).
    center_xy : (x0, y0)
        Center in pixel coordinates.
    pa_deg, inclination_deg : float
        Fixed geometry in the convention documented in :func:`sky_to_disk`.
    ring_edges : 1-D array
        Increasing radial edges in pixels.
    valid_mask : 2-D bool array, optional
        Additional user-defined mask.

    Returns a list of dictionaries. Uncertainties are not reported yet because
    spatially correlated pixels make naive least-squares covariance optimistic.
    """
    vmap = np.asarray(velocity_map, dtype=float)
    if vmap.ndim != 2:
        raise ValueError("velocity_map must be 2-D")
    edges = np.asarray(ring_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("ring_edges must be a strictly increasing 1-D array")
    y, x = np.indices(vmap.shape)
    x0, y0 = center_xy
    _, _, radius, cos_theta = sky_to_disk(x, y, x0, y0, pa_deg, inclination_deg)
    base = np.isfinite(vmap)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != vmap.shape:
            raise ValueError("valid_mask must match velocity_map shape")
        base &= valid_mask
    sin_i = np.sin(np.deg2rad(inclination_deg))
    if abs(sin_i) < 1e-3:
        raise ValueError("Near face-on geometry cannot constrain rotation speed reliably")

    results = []
    for r0, r1 in zip(edges[:-1], edges[1:]):
        sel = base & (radius >= r0) & (radius < r1) & (np.abs(cos_theta) >= 0.05)
        n = int(sel.sum())
        if n < min_pixels:
            continue
        c = sin_i * cos_theta[sel]
        v = vmap[sel]
        initial_vsys = float(np.nanmedian(v))
        initial_vrot = float(np.nanmedian((v - initial_vsys) / c))
        initial_vrot = float(np.clip(initial_vrot, 0, 500))
        fit = least_squares(lambda p: p[0] * c + p[1] - v,
                            x0=[initial_vrot, initial_vsys],
                            bounds=([0.0, -np.inf], [600.0, np.inf]),
                            loss="soft_l1")
        residual = v - (fit.x[0] * c + fit.x[1])
        results.append({
            "r_inner_pix": float(r0), "r_outer_pix": float(r1),
            "radius_mid_pix": float((r0 + r1) / 2),
            "vrot_kms": float(fit.x[0]), "vsys_kms": float(fit.x[1]),
            "residual_rms_kms": float(np.sqrt(np.mean(residual**2))),
            "n_pixels": n, "success": bool(fit.success),
        })
    return results
