"""Peak traced bytes of one L2 normalize of the MovieLens-1M rating matrix.

Each row is one user's dense rating vector (zeros where the user did not
rate). The matrix is built before tracing, so the peak is memory allocated
inside normalize itself. Stdout is a single integer: peak traced bytes.
"""

import contextlib
import gc
import io
import os
import tracemalloc

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from cornac.data import Dataset
from cornac.datasets import movielens
from cornac.utils.common import normalize


def main():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        data = movielens.load_feedback(variant="1M")
        dataset = Dataset.from_uir(data)
        dense = np.asarray(dataset.csr_matrix.toarray(), dtype=np.float64)
    if dense.shape[0] < 6000 or dense.shape[1] < 3000:
        raise SystemExit(f"unexpected rating-matrix shape {dense.shape}")
    gc.collect()
    tracemalloc.start()
    normalized = normalize(dense, norm="l2", axis=1, copy=True)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if normalized.shape != dense.shape or not np.isfinite(normalized).all():
        raise SystemExit("normalize did not return a finite matrix")
    # Keep the result alive until the peak has been read.
    if normalized is None:
        raise SystemExit("missing normalized matrix")
    print(peak)


if __name__ == "__main__":
    main()
