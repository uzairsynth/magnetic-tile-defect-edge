import numpy as np

from tiles.metrics import choose_defect_threshold, defect_alarm, defect_recall, group_bootstrap, macro_f1


def noisy_probs(y, rng, noise=0.6):
    logits = np.eye(6)[y] * 2 + rng.normal(0, noise, (len(y), 6))
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True)


def test_defect_threshold_reaches_target_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 6, 400)
    probs = noisy_probs(y, rng, noise=1.2)
    t = choose_defect_threshold(y, probs, 0.98)
    assert defect_alarm(y, probs, t)["recall"] >= 0.98
    # Any stricter threshold drops below the target, so t raises the fewest alarms.
    assert defect_alarm(y, probs, t + 1e-6)["recall"] < 0.98


def test_group_bootstrap_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    y = np.repeat(rng.integers(0, 6, 60), 3)  # 60 tiles, three views each
    groups = np.repeat(np.arange(60), 3)
    probs = noisy_probs(y, rng)
    lo, hi = group_bootstrap(y, probs, groups, macro_f1, n=300)
    assert lo <= macro_f1(y, probs) <= hi
    lo_r, hi_r = group_bootstrap(y, probs, groups, defect_recall, n=300)
    assert 0 <= lo_r <= hi_r <= 1
