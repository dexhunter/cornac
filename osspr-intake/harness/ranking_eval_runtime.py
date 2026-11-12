"""Time one base_method.ranking_eval pass on MovieLens-1M.

MostPop scores every test user and Recall@10 is the metric, matching a
normal cornac experiment with the library default rating threshold.
Stdout is a single number: elapsed milliseconds of ranking_eval only.
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

from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.eval_methods.base_method import ranking_eval
from cornac.metrics import Recall
from cornac.models import MostPop


def main():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        data = movielens.load_feedback(variant="1M")
        split = RatioSplit(
            data,
            test_size=0.2,
            rating_threshold=1.0,
            seed=42,
            exclude_unknowns=True,
            verbose=False,
        )
        model = MostPop()
        model.fit(split.train_set)
    gc.collect()
    metric = Recall(k=10)
    started = time.perf_counter()
    averages, per_user = ranking_eval(
        model,
        [metric],
        split.train_set,
        split.test_set,
        rating_threshold=1.0,
        exclude_unknowns=True,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if len(per_user[0]) < 5000 or not (averages[0] == averages[0]):
        raise SystemExit("ranking_eval did not cover the MovieLens-1M test users")
    print(f"{elapsed_ms:.3f}")


if __name__ == "__main__":
    main()
