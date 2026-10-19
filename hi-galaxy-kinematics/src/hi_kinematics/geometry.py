"""Simple image-plane utilities; these do not deproject a spectral cube."""
import numpy as np
from skimage.measure import EllipseModel


def estimate_flux_center(moment0):
    """Flux-weighted center using finite, non-negative intensity values."""
    image = np.asarray(moment0, dtype=float)
    if image.ndim != 2:
        raise ValueError("moment0 must be 2-D")
    weights = np.where(np.isfinite(image) & (image > 0), image, 0.0)
    total = weights.sum()
    if total <= 0:
        raise ValueError("Cannot estimate center: no positive finite flux")
    y, x = np.indices(image.shape)
    return float((x * weights).sum() / total), float((y * weights).sum() / total)


def fit_outer_ellipse(moment0, threshold_fraction=0.5):
    """Fit an ellipse to pixels above a fraction of the peak intensity.

    Returns (x_center, y_center, semi_axis_a, semi_axis_b, theta_rad). The
    angle follows scikit-image's EllipseModel convention and is not guaranteed
    to equal a survey's astronomical position angle without convention checks.
    """
    image = np.asarray(moment0, dtype=float)
    good = np.isfinite(image)
    if not good.any():
        raise ValueError("moment0 has no finite pixels")
    peak = np.nanmax(image)
    if peak <= 0:
        raise ValueError("moment0 has no positive peak")
    yy, xx = np.indices(image.shape)
    mask = good & (image > threshold_fraction * peak)
    points = np.column_stack((xx[mask], yy[mask]))
    if len(points) < 5:
        raise ValueError("Too few pixels for ellipse fit")
    model = EllipseModel()
    if not model.estimate(points):
        raise RuntimeError("Ellipse fit failed")
    return tuple(float(v) for v in model.params)
