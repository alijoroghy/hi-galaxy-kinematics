from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from hi_kinematics.cube import load_fits_cube, spectral_axis_kms, moment_maps
from hi_kinematics import pipeline




def main():
    print("Loading FITS cube...")

    cube, header = load_fits_cube(pipeline.ACTIVE_CUBE_PATH)
    velocity = spectral_axis_kms(header, cube.shape[0])

    print(f"Cube shape: {cube.shape}")
    print(f"Velocity range: {velocity.min():.3f} to {velocity.max():.3f} km/s")
    print(f"Median channel spacing: {np.median(np.diff(velocity)):.6f} km/s")

    print("\nChecking channel spacing...")
    channel_differences = np.diff(velocity)
    print(f"Minimum channel spacing: {channel_differences.min():.6f} km/s")
    print(f"Maximum channel spacing: {channel_differences.max():.6f} km/s")

    print("\nChecking cube values...")
    finite = np.isfinite(cube)
    print(f"Finite fraction: {finite.mean():.6f}")
    print(f"Minimum finite intensity: {np.nanmin(cube):.6g}")
    print(f"Maximum finite intensity: {np.nanmax(cube):.6g}")
    print(f"Median intensity: {np.nanmedian(cube):.6g}")

    print("\nCalculating preliminary moment maps...")
    m0, m1 = moment_maps(cube, velocity)

    print(f"Moment 0 finite pixels: {np.isfinite(m0).sum()}")
    print(f"Moment 1 finite pixels: {np.isfinite(m1).sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    image0 = axes[0].imshow(m0, origin="lower", cmap="inferno")
    axes[0].set_title("Preliminary Moment 0")
    axes[0].set_xlabel("X pixel")
    axes[0].set_ylabel("Y pixel")
    fig.colorbar(image0, ax=axes[0], label="Integrated intensity")

    image1 = axes[1].imshow(m1, origin="lower", cmap="RdBu_r")
    axes[1].set_title("Preliminary Moment 1")
    axes[1].set_xlabel("X pixel")
    axes[1].set_ylabel("Y pixel")
    fig.colorbar(image1, ax=axes[1], label="Velocity (km/s)")

    fig.tight_layout()

    output_path = Path("preliminary_moment_maps.png")
    fig.savefig(output_path, dpi=160)
    print(f"\nSaved figure: {output_path.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()