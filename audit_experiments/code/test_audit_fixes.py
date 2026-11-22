"""Tests for the 22 Nov 2026 audit fixes. Run: python3 -m pytest -q code/test_audit_fixes.py"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _exact_ap():
    src = (Path(__file__).resolve().parent / 'p5_analyze.py').read_text()
    i, j = src.index('def exact_ap'), src.index('\ndef summarise')
    ns = {'np': np}
    exec(src[i:j], ns)
    return ns['exact_ap']


def test_ap_ties_do_not_depend_on_row_order():
    ap = _exact_ap()
    assert ap(np.array([.5, .5]), np.array([1, 0])) == ap(np.array([.5, .5]), np.array([0, 1])) == 0.5


def test_ap_matches_sklearn_with_many_ties():
    from sklearn.metrics import average_precision_score
    ap, rng = _exact_ap(), np.random.default_rng(1)
    for _ in range(20):
        s, y = rng.integers(0, 15, 400).astype(float), rng.random(400) < 0.25
        assert abs(ap(s, y) - average_precision_score(y, s)) < 1e-12


def test_one_percent_threshold():
    src = (Path(__file__).resolve().parent / 'p5_analyze.py').read_text()
    assert 'ONE_PCT = 656' in src and int(np.ceil(0.01 * 256 * 256)) == 656


def test_unit_seeds_are_stable_and_distinct():
    import p5_core as C
    seeds = [C.unit_seed(c) for c in ('candle', 'cashew', 'fryum', 'macaroni2', 'pcb2', 'pcb4')]
    assert len(set(seeds)) == len(seeds) and C.unit_seed('candle') == seeds[0]
    a, b = C.bootstrap_counts(40, reps=50, seed=seeds[0]), C.bootstrap_counts(40, reps=50, seed=seeds[1])
    assert not np.array_equal(a, b)
