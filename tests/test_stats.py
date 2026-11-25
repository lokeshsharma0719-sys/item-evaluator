from __future__ import annotations

import pytest

from item_metrics import normalise as N
from item_metrics.stats import bootstrap, cohen_kappa, fleiss_kappa, krippendorff_alpha_nominal, prop, wilson


def test_wilson_known_values():
    lo, hi = wilson(5, 10)
    assert (round(lo, 4), round(hi, 4)) == (0.2366, 0.7634)
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and round(hi, 4) == 0.2775
    lo, hi = wilson(10, 10)
    assert round(lo, 4) == 0.7225 and hi == pytest.approx(1.0)
    assert prop(0, 0)["value"] is None
    assert prop(3, 5)["small_n"] is True


def test_kappas():
    a = ["x", "y", "x", "z"]
    assert cohen_kappa(a, a) == 1.0
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0
    assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(-1.0)
    assert fleiss_kappa([["a", "a", "a"], ["b", "b", "b"]]) == 1.0


def test_krippendorff_alpha_with_missing():
    units = [["a", "a"], ["b", "b"], ["a", "b"], ["a", None]]   # last unit has one value: not pairable
    assert krippendorff_alpha_nominal(units) == pytest.approx(1 - 5 * 2 / 18)
    assert krippendorff_alpha_nominal([["a", "a", None], ["b", "b", "b"]]) == 1.0


def test_bootstrap_is_seeded():
    xs = [0, 1, 1, 0, 1, 1, 1, 0]
    f = lambda s: sum(s) / len(s)
    assert bootstrap(xs, f, seed=7) == bootstrap(xs, f, seed=7)


def test_normalisation_rules():
    assert N.agreement(-0.3) == 0.0
    assert N.agreement(0.7) == 0.7
    assert N.lower_better(0.1) == pytest.approx(0.9)
    assert N.chance_adjusted(-0.4) == 0.0
    assert N.error_score(0.06) == pytest.approx(0.94)


def test_composite():
    w = {"kcr": 0.5, "per": 0.25, "no_ambiguity": 0.25}
    v, used = N.composite({"kcr": 0.9, "per": 0.8, "no_ambiguity": 0.9}, w, {"kcr"})
    assert round(v, 3) == 0.874
    assert round(0.9 ** 0.5 * 0.8 ** 0.25 * 0.9 ** 0.25, 3) == round(v, 3)
    v, used = N.composite({"kcr": 0.9, "per": None, "no_ambiguity": 0.9}, w, {"kcr"})
    assert used == {"kcr": pytest.approx(0.6667, abs=1e-4), "no_ambiguity": pytest.approx(0.3333, abs=1e-4)}
    assert v == pytest.approx(0.9)
    v, used = N.composite({"kcr": None, "per": 0.8, "no_ambiguity": 0.9}, w, {"kcr"})
    assert v is None
    v, _ = N.composite({"kcr": 0.0, "per": 1.0, "no_ambiguity": 1.0}, w, {"kcr"})
    assert v == pytest.approx(0.01 ** 0.5)   # floored, and pulled down sharply
