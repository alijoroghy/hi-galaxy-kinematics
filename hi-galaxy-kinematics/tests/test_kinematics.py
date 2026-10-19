import numpy as np
from hi_kinematics.kinematics import rotation_model, fit_rotation_curve
from hi_kinematics.geometry import estimate_flux_center


def test_rotation_curve_recovers_synthetic_speed():
    rng = np.random.default_rng(3)
    y, x = np.indices((120, 120))
    center = (59.5, 60.2)
    pa, inc, vrot, vsys = 25.0, 55.0, 125.0, 80.0
    model = rotation_model(x, y, *center, pa, inc, vrot, vsys)
    noisy = model + rng.normal(0, 1.0, model.shape)
    rows = fit_rotation_curve(noisy, center, pa, inc, [10, 25, 40, 55], min_pixels=20)
    assert len(rows) >= 2
    assert abs(np.median([r["vrot_kms"] for r in rows]) - vrot) < 8
    assert abs(np.median([r["vsys_kms"] for r in rows]) - vsys) < 3


def test_flux_center():
    image = np.zeros((11, 13))
    image[4, 8] = 10
    assert estimate_flux_center(image) == (8.0, 4.0)
