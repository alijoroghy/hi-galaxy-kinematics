from __future__ import annotations

import numpy as np
import pywt

from scipy import ndimage


def estimate_robust_noise(data):
    """
    Estimate noise robustly using the MAD estimator.
    """
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("No finite values available for noise estimation.")

    median = np.median(values)
    mad = np.median(np.abs(values - median))

    sigma = mad / 0.67448975

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Could not estimate a positive noise level.")

    return float(sigma)


def create_signal_mask(
    cube,
    threshold_sigma=4.0,
    grow_sigma=2.0,
    spatial_grow=2,
):
    """
    Create a 3D signal mask for a spectral cube.

    Initial detection:
        |data - channel_median| >= threshold_sigma * channel_noise

    Mask growth:
        Expand detected emission spatially and spectrally, retaining
        weaker neighboring voxels above grow_sigma * channel_noise.

    Returns
    -------
    mask : 3D bool ndarray
        True means likely signal; False means candidate noise.
    """
    cube = np.asarray(cube, dtype=float)

    if cube.ndim != 3:
        raise ValueError("Expected cube shape (spectral, y, x).")

    nchan, ny, nx = cube.shape

    residual = np.full(cube.shape, np.nan, dtype=float)
    noise = np.zeros(nchan, dtype=float)

    # Estimate baseline and noise separately for every channel.
    for k in range(nchan):
        plane = cube[k]
        finite = np.isfinite(plane)

        if finite.sum() == 0:
            continue

        baseline = np.median(plane[finite])
        residual[k] = plane - baseline

        noise[k] = estimate_robust_noise(residual[k])

    valid_noise = noise[np.isfinite(noise) & (noise > 0)]

    if valid_noise.size == 0:
        raise ValueError("Could not estimate channel noise.")

    # Replace unusable channel estimates with the median noise.
    fallback = float(np.median(valid_noise))
    noise[~np.isfinite(noise) | (noise <= 0)] = fallback

    noise_cube = noise[:, None, None]

    # Initial high-confidence detections.
    strong = (
        np.isfinite(residual)
        & (np.abs(residual) >= threshold_sigma * noise_cube)
    )

    # Weaker voxels that may belong to extended emission.
    weak = (
        np.isfinite(residual)
        & (np.abs(residual) >= grow_sigma * noise_cube)
    )

    # Grow the mask spatially and spectrally.
    structure = np.ones((3, 3, 3), dtype=bool)

    expanded = ndimage.binary_dilation(
        strong,
        structure=structure,
        iterations=spatial_grow,
    )

    mask = expanded & weak

    return mask, noise, residual


def wavelet_denoise_3d(
    data,
    wavelet="db2",
    level=1,
    threshold_factor=2.5,
    threshold_sigma=4.0,
    grow_sigma=2.0,
    spatial_grow=2,
):
    """
    Detect likely signal, then strongly denoise only outside the signal mask.

    Signal voxels are preserved exactly.
    Candidate-noise voxels are wavelet-denoised.

    Parameters
    ----------
    data : ndarray
        Input cube shaped (spectral, y, x).
    wavelet : str
        Wavelet family.
    level : int
        Wavelet decomposition level.
    threshold_factor : float
        Multiplier applied to the estimated wavelet noise threshold.
        Larger values produce stronger denoising.
    threshold_sigma : float
        High-confidence signal-detection threshold.
    grow_sigma : float
        Lower threshold for retaining neighboring emission.
    spatial_grow : int
        Number of 3D mask-growth iterations.

    Returns
    -------
    denoised : ndarray
        Denoised cube with the original shape.
    mask : ndarray
        Boolean signal mask; True denotes likely signal.
    """
    data = np.asarray(data, dtype=float)

    if data.ndim != 3:
        raise ValueError("Input data must be a 3D cube.")

    if not np.all(np.isfinite(data)):
        raise ValueError(
            "Input contains NaN or infinite values. "
            "Handle invalid voxels before denoising."
        )

    if threshold_factor <= 0:
        raise ValueError("threshold_factor must be positive.")

    # 1. Identify likely signal.
    mask, channel_noise, residual = create_signal_mask(
        data,
        threshold_sigma=threshold_sigma,
        grow_sigma=grow_sigma,
        spatial_grow=spatial_grow,
    )

    # 2. Keep original signal unchanged.
    denoised = data.copy()

    # If no noise candidates exist, return safely.
    noise_mask = ~mask

    if not np.any(noise_mask):
        return denoised, mask

    # 3. Wavelet decomposition of the residual cube.
    # Remove each channel's baseline before denoising.
    channel_baseline = np.nanmedian(data, axis=(1, 2))
    baseline_cube = channel_baseline[:, None, None]

    centered = data - baseline_cube

    wavelet_obj = pywt.Wavelet(wavelet)

    max_level = pywt.dwtn_max_level(centered.shape, wavelet_obj)

    if level < 1 or level > max_level:
        raise ValueError(
            f"Invalid level={level}. Maximum allowed level is {max_level}."
        )

    coeffs = pywt.wavedecn(
        centered,
        wavelet=wavelet_obj,
        level=level,
        mode="symmetric",
    )

    # Estimate noise from the finest detail coefficients.
    finest_details = coeffs[-1]

    detail_values = np.concatenate([
        values.ravel()
        for values in finest_details.values()
    ])

    coeff_median = np.median(detail_values)

    sigma_wavelet = (
        np.median(np.abs(detail_values - coeff_median))
        / 0.67448975
    )

    if not np.isfinite(sigma_wavelet) or sigma_wavelet <= 0:
        sigma_wavelet = float(np.median(channel_noise))

    threshold = (
        threshold_factor
        * sigma_wavelet
        * np.sqrt(2 * np.log(centered.size))
    )

    # Keep approximation coefficients unchanged.
    filtered_coeffs = [coeffs[0]]

    for detail_level in coeffs[1:]:
        filtered_coeffs.append({
            key: pywt.threshold(values, threshold, mode="soft")
            for key, values in detail_level.items()
        })

    reconstructed = pywt.waverecn(
        filtered_coeffs,
        wavelet=wavelet_obj,
        mode="symmetric",
    )

    reconstructed = reconstructed[
        :data.shape[0],
        :data.shape[1],
        :data.shape[2],
    ]

    reconstructed += baseline_cube

    # 4. Replace only voxels outside the signal mask.
    denoised[noise_mask] = reconstructed[noise_mask]

    return denoised, mask