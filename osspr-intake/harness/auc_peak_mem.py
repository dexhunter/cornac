"""Peak traced bytes while AUC.compute scores every MovieLens-1M test user.

Candidates match base_method.ranking_eval at the default rating threshold:
test items are positives, training items are removed, and the remaining
catalog is scored with MostPop. Inputs are built before tracing, so the
peak is the memory AUC.compute itself allocates (the pairwise score tiles).
Stdout is a single integer: peak traced bytes.
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

from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.metrics import AUC
from cornac.models import MostPop


def _eval_inputs(train_set, test_set, item_pop, rating_threshold):
    test_mat = test_set.csr_matrix
    train_mat = train_set.csr_matrix
    prepared = []
    seen = set()
    for user_idx in test_set.uir_tuple[0]:
        user_idx = int(user_idx)
        if user_idx in seen:
            continue
        seen.add(user_idx)
        test_row = test_mat.getrow(user_idx)
        test_pos = [
            int(item_idx)
            for item_idx, rating in zip(test_row.indices, test_row.data)
            if rating >= rating_threshold
        ]
        if not test_pos:
            continue
        pos_mask = np.zeros(test_set.num_items, dtype=np.int32)
        neg_mask = np.ones(test_set.num_items, dtype=np.int32)
        pos_mask[test_pos] = 1
        train_pos = []
        if user_idx < train_mat.shape[0]:
            train_row = train_mat.getrow(user_idx)
            train_pos = [
                int(item_idx)
                for item_idx, rating in zip(train_row.indices, train_row.data)
                if rating >= rating_threshold
            ]
        neg_mask[test_pos + train_pos] = 0
        pos_mask = pos_mask[: train_set.num_items]
        neg_mask = neg_mask[: train_set.num_items]
        item_indices = np.nonzero(pos_mask + neg_mask)[0]
        gt_pos = np.nonzero(pos_mask)[0]
        gt_neg = np.nonzero(neg_mask)[0]
        prepared.append((item_indices, item_pop[item_indices], gt_pos, gt_neg))
    return prepared


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
        item_pop = np.asarray(model.score(0), dtype=np.float64)
        prepared = _eval_inputs(
            split.train_set, split.test_set, item_pop, rating_threshold=1.0
        )
    if len(prepared) < 5000:
        raise SystemExit("too few MovieLens-1M test users")
    gc.collect()
    auc = AUC()
    tracemalloc.start()
    total = 0.0
    for item_indices, scores, gt_pos, gt_neg in prepared:
        total += auc.compute(
            item_indices=item_indices,
            pd_scores=scores,
            gt_pos=gt_pos,
            gt_neg=gt_neg,
        )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if not (total == total) or total <= 0:
        raise SystemExit("AUC scores were not finite")
    print(peak)


if __name__ == "__main__":
    main()
