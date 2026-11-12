"""Time one Reader.read of the cached MovieLens-1M ratings file.

The timed call is the second read in this process so the page cache is warm.
Stdout is a single number: elapsed milliseconds.
"""

import contextlib
import gc
import io
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from cornac.data import Reader
from cornac.datasets.movielens import ML_DATASETS
from cornac.utils import cache


def _ratings_path():
    ml = ML_DATASETS["1M"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return cache(url=ml.url, unzip=ml.unzip, relative_path=ml.path), ml.sep


def main():
    fpath, sep = _ratings_path()
    reader = Reader()
    reader.read(fpath, fmt="UIR", sep=sep)
    gc.collect()
    started = time.perf_counter()
    rows = reader.read(fpath, fmt="UIR", sep=sep)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if len(rows) != 1000209 or rows[0] != ("1", "1193", 5.0):
        raise SystemExit("unexpected MovieLens-1M parse")
    print(f"{elapsed_ms:.3f}")


if __name__ == "__main__":
    main()
