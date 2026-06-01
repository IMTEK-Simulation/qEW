"""Augment a C++ qew_static HDF5 file with the analysis datasets plot.py reads.

The C++ executable writes only the relaxed profiles `h` (analysis stays in
Python, per the port plan). This script computes the height-difference
autocorrelation, PSD and distance axis for every `.../h` dataset using the
validated routines in ../../Overleaf/qEW/qew_analysis.py, and writes
`ell`, `acf`, `q`, `psd` alongside each `h`. The result is readable by
../../Overleaf/qEW/plot.py unchanged.

Usage:
    python3 postprocess_static.py [static_arclength.h5 ...]
"""

import os
import sys

import h5py
import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                    "Overleaf", "qEW")
)
from qew_analysis import height_difference_acf, psd as compute_psd  # noqa: E402


def process(fname):
    with h5py.File(fname, "a") as hf:
        L = float(hf["physical_size"][()])

        targets = []
        hf.visititems(
            lambda name, obj: targets.append(name)
            if isinstance(obj, h5py.Dataset) and name.endswith("/h")
            else None
        )

        for name in targets:
            grp = hf[name].parent
            h = hf[name][:]
            ell, acf = height_difference_acf(h, L)
            q, C = compute_psd(h, L)
            for key, val in (("ell", ell), ("acf", acf), ("q", q), ("psd", C)):
                if key in grp:
                    del grp[key]
                grp.create_dataset(key, data=val)
        print(f"augmented {fname}: {len(targets)} profile(s)")


if __name__ == "__main__":
    files = sys.argv[1:] or ["static_arclength.h5"]
    for f in files:
        process(f)
