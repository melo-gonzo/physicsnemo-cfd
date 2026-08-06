# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the conformal *diagnostic* sample metric (approximate resplit coverage/width).

Guards the correctness properties of the demoted diagnostic: a sketch whose lower-bound CDF read
never under-states miscoverage (conservative by construction), a **leak-free** per-split
calibration grid, ``NaN`` (never a silently-maxed lambda) on infeasible splits, non-finite
case-fields skipped so a NaN case can never inflate reported coverage, the unified per-field
``amax`` risk event, and per-run knobs (``alpha`` / ``n_splits`` / ``test_frac`` / ``sketch_q`` /
``sigma_eps``). The companion ``conformal_analysis.py`` is covered core-free (its core-lib
imports are lazy): split guards, export loading with per-dataset grouping, missing-``dataset``-key
fallback, std omission, and field aliasing; only the end-to-end sweep through the core calibrator
stays behind ``importorskip``. This is a diagnostic, not a guaranteed artifact — the
authoritative guarantee is the library-backed conformal report / engine-integrated calibrator.
"""

from __future__ import annotations

import importlib.util
import logging
import pathlib

import numpy as np
import pytest

from physicsnemo.cfd.evaluation.datasets.schema import FieldDistribution
from physicsnemo.cfd.evaluation.metrics import get_metric
from physicsnemo.cfd.evaluation.metrics.builtin.conformal import (
    _DEFAULT_SKETCH_Q,
    _ConformalDiagnostic,
    _crc_lambda,
    _loss_on_grid,
    _sketch,
    register_conformal_metrics,
)


def _halfnormal_sketches(n, npts, scale, seed=1):
    """Stack ``n`` conservative sketches of half-normal |scores| (npts each)."""
    rng = np.random.default_rng(seed)
    return np.stack([_sketch(np.abs(rng.normal(0.0, scale, npts))) for _ in range(n)])


def _load_companion():
    """Import ``conformal_analysis.py`` by path (no core-lib requirement: its
    ``physicsnemo.experimental.uq.conformal`` imports are lazy, inside the scoring functions).
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    path = root / "workflows" / "benchmarking" / "conformal_analysis.py"
    spec = importlib.util.spec_from_file_location("conformal_analysis", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _collect(stats_list):
    """Mimic the engine's per-case collection of ``partial`` outputs into columns."""
    collected: dict[str, list] = {}
    for stats in stats_list:
        for k, v in stats.items():
            collected.setdefault(k, []).append(v)
    return collected


def test_sketch_is_conservative_upper_quantile():
    """The upper-quantile sketch never sits below the linear-interpolation quantile."""
    rng = np.random.default_rng(0)
    s = np.abs(rng.normal(0.0, 1.0, 2000))
    sk = _sketch(s)
    assert np.all(sk >= np.quantile(s, np.linspace(0.0, 1.0, _DEFAULT_SKETCH_Q)) - 1e-9)
    assert sk.shape == (_DEFAULT_SKETCH_Q,)


def test_sketch_read_never_understates_miscoverage_atom_counterexample():
    """One tiny score among 10k huge ones: the sketch-read miscoverage at λ=0 must not
    under-state the true 0.9999 (the old ``m/Q`` read gave ~0.99225)."""
    scores = np.full(10000, 100.0)
    scores[0] = 0.0
    sk = _sketch(scores)
    true_misc = float(np.mean(scores > 0.0))  # 0.9999
    read = float(_loss_on_grid(sk, np.array([0.0]))[0])
    assert read >= true_misc
    # Below the sketch minimum nothing is certified: the loss clamps to 1.
    assert float(_loss_on_grid(sk, np.array([-1.0]))[0]) == 1.0


def test_sketch_read_is_conservative_on_a_grid():
    """Property check: the sketch-read miscoverage upper-bounds the true empirical miscoverage
    at every λ, over-stating it by at most ~1/(Q-1) plus quantile granularity."""
    rng = np.random.default_rng(3)
    s = np.abs(rng.normal(0.0, 1.0, 5000))
    sk = _sketch(s)
    grid = np.linspace(-0.5, 4.0, 301)
    true = np.array([np.mean(s > lam) for lam in grid])
    read = _loss_on_grid(sk, grid)
    assert np.all(read >= true - 1e-12)
    assert float(np.max(read - true)) <= 2.0 / (_DEFAULT_SKETCH_Q - 1)


def test_crc_lambda_returns_nan_when_infeasible():
    """Scores all above the grid leave no feasible candidate -> NaN, not a maxed lambda."""
    sketches = np.stack([_sketch(np.full(100, 5.0)) for _ in range(3)])
    lam = _crc_lambda(sketches, np.linspace(0.0, 1.0, 50), alpha=0.1)
    assert np.isnan(lam)


def test_eval_arm_is_conservatively_calibrated():
    """Resplit coverage sits at/above the 1-alpha target with a positive width."""
    d = _ConformalDiagnostic(alpha=0.1)
    sk = _halfnormal_sketches(200, 400, 1.0)
    cov, wid = d._eval_arm(sk, sig=None, rng=np.random.default_rng(0), alpha=0.1)
    assert np.isfinite(cov) and np.isfinite(wid)
    assert cov >= 0.88
    assert wid > 0.0


def test_eval_arm_nan_on_infeasible_split():
    """Too few cases (1/(n_cal+1) > alpha) yield NaN, not a silently-maxed lambda."""
    d = _ConformalDiagnostic(alpha=0.1)
    sk = _halfnormal_sketches(6, 200, 1.0)
    cov, wid = d._eval_arm(sk, sig=None, rng=np.random.default_rng(0), alpha=0.1)
    assert np.isnan(cov) and np.isnan(wid)


def test_alpha_is_configurable_and_monotone():
    """A higher miscoverage target lowers coverage and tightens the band."""
    d = _ConformalDiagnostic(alpha=0.1)
    sk = _halfnormal_sketches(200, 400, 1.0)
    _, wid_10 = d._eval_arm(sk, sig=None, rng=np.random.default_rng(0), alpha=0.1)
    cov_20, wid_20 = d._eval_arm(sk, sig=None, rng=np.random.default_rng(0), alpha=0.2)
    assert cov_20 < 0.9
    assert wid_20 < wid_10


def test_eval_arm_is_deterministic_given_seed():
    """The same seed reproduces the resplit coverage/width exactly."""
    d = _ConformalDiagnostic(alpha=0.1)
    sk = _halfnormal_sketches(120, 300, 1.0)
    a = d._eval_arm(sk, sig=None, rng=np.random.default_rng(0), alpha=0.1)
    b = d._eval_arm(sk, sig=None, rng=np.random.default_rng(0), alpha=0.1)
    assert a == b


def test_finalize_emits_diag_keys_and_threads_alpha():
    """finalize emits per-field + mean diag keys, threads alpha, and never claims a guarantee."""
    d = _ConformalDiagnostic(alpha=0.1)
    sk = _halfnormal_sketches(80, 300, 1.0)
    collected: dict[str, list] = {"_alpha": [0.1] * 80}
    for g in range(_DEFAULT_SKETCH_Q):
        collected[f"pressure::c{g}"] = list(sk[:, g])
    out = d.finalize_samples(collected)
    assert "pressure_diag_coverage_const" in out
    assert "pressure_diag_width_const" in out
    assert "mean_diag_coverage_const" in out
    assert out["diag_alpha"] == 0.1
    assert not any("crc" in k for k in out)
    assert np.isfinite(out["pressure_diag_coverage_const"])


def test_partial_unifies_vector_field_amax():
    """A vector field yields ONE per-field risk (amax over components), not one per component."""
    rng = np.random.default_rng(0)
    n = 64
    true = rng.normal(0.0, 1.0, (n, 3))
    pred = true.copy()
    pred[:, 1] += 3.0  # only the y-component is badly wrong
    d = _ConformalDiagnostic(alpha=0.1)
    stats = d.partial(
        {"wss": true},
        {"wss": FieldDistribution(mean=pred, std=np.ones((n, 3)))},
    )
    fields = {k.split("::")[0] for k in stats if "::" in k}
    assert fields == {"wss"}  # not wss_x / wss_y / wss_z
    expected = _sketch(np.abs(true - pred).max(axis=1))
    got = np.array([stats[f"wss::c{g}"] for g in range(_DEFAULT_SKETCH_Q)])
    assert np.allclose(got, expected)


def test_nan_case_cannot_inflate_coverage(caplog):
    """An all-NaN prediction case is skipped with a warning and must NOT raise diag_coverage
    (pre-fix a NaN sketch on the test side counted as 100% covered: ``NaN > λ`` is False).
    """
    rng = np.random.default_rng(0)
    d = _ConformalDiagnostic(alpha=0.1, n_splits=10)
    clean = []
    for _ in range(30):
        true = rng.normal(0.0, 1.0, 200)
        pred = true + rng.normal(0.0, 0.5, 200)
        clean.append(
            d.partial({"pressure": true}, {"pressure": FieldDistribution(mean=pred)})
        )
    base = d.finalize_samples(_collect(clean))
    with caplog.at_level(logging.WARNING):
        nan_stats = d.partial(
            {"pressure": rng.normal(0.0, 1.0, 200)},
            {"pressure": FieldDistribution(mean=np.full(200, np.nan))},
        )
    assert any("non-finite" in rec.getMessage() for rec in caplog.records)
    assert not any("::" in k for k in nan_stats)  # the case-field was skipped entirely
    out = d.finalize_samples(_collect(clean + [nan_stats]))
    assert np.isfinite(out["pressure_diag_coverage_const"])
    assert out["pressure_diag_coverage_const"] <= base["pressure_diag_coverage_const"]


def test_nonfinite_std_drops_adaptive_arm_only(caplog):
    """A non-finite std drops only the adaptive arm for that case; the const arm survives."""
    d = _ConformalDiagnostic(alpha=0.1)
    true = np.zeros(50)
    pred = np.ones(50)
    sig = np.ones(50)
    sig[3] = np.nan
    with caplog.at_level(logging.WARNING):
        stats = d.partial(
            {"pressure": true},
            {"pressure": FieldDistribution(mean=pred, std=sig)},
        )
    assert any(k.startswith("pressure::c") for k in stats)
    assert not any(k.startswith("pressure::a") for k in stats)
    assert any(
        "non-finite or negative std" in rec.getMessage() for rec in caplog.records
    )


def test_partial_threads_run_overrides():
    """Per-run alpha / n_splits / test_frac / sketch_q / sigma_eps flow through partial."""
    d = _ConformalDiagnostic(alpha=0.1)
    true = np.zeros(30)
    pred = np.ones(30)
    stats = d.partial(
        {"pressure": true},
        {"pressure": FieldDistribution(mean=pred, std=np.ones(30))},
        alpha=0.2,
        n_splits=7,
        test_frac=0.3,
        sketch_q=17,
        sigma_eps=1e-6,
    )
    assert stats["_alpha"] == 0.2
    assert stats["_n_splits"] == 7
    assert stats["_test_frac"] == 0.3
    assert stats["_sketch_q"] == 17
    assert sum(1 for k in stats if k.startswith("pressure::c")) == 17


def test_sketch_q_override_roundtrips_through_finalize():
    """finalize reconstructs sketches at the per-run sketch_q / n_splits stashed by partial."""
    d = _ConformalDiagnostic(alpha=0.1)
    sk = _halfnormal_sketches(60, 200, 1.0)
    q = _DEFAULT_SKETCH_Q
    collected: dict[str, list] = {
        "_alpha": [0.1] * 60,
        "_sketch_q": [float(q)] * 60,
        "_n_splits": [8.0] * 60,
        "_test_frac": [0.4] * 60,
    }
    for g in range(q):
        collected[f"pressure::c{g}"] = list(sk[:, g])
    out = d.finalize_samples(collected)
    assert np.isfinite(out["pressure_diag_coverage_const"])


def test_registered_under_diagnostic_and_legacy_alias():
    """The metric registers under conformal_diagnostic with conformal_crc as a live alias."""
    register_conformal_metrics(alpha=0.1)
    for domain in ("surface", "volume"):
        assert get_metric("conformal_diagnostic", domain=domain) is not None
        assert get_metric("conformal_crc", domain=domain) is get_metric(
            "conformal_diagnostic", domain=domain
        )


def test_register_accepts_run_param_defaults():
    """register_conformal_metrics threads the new knob defaults onto the metric instance."""
    register_conformal_metrics(
        alpha=0.05, n_splits=12, test_frac=0.25, sketch_q=65, sigma_eps=1e-6
    )
    m = get_metric("conformal_diagnostic", domain="surface")
    assert m.alpha == 0.05 and m.n_splits == 12 and m.test_frac == 0.25
    assert m.sketch_q == 65 and m.sigma_eps == 1e-6
    # Restore the module default registration for any later tests.
    register_conformal_metrics(alpha=0.1)


def test_companion_default_ncal_grid_leaves_holdout():
    """The companion's default n_cal grid never empties the held-out set (runs without the
    core conformal lib: the companion imports it lazily)."""
    m = _load_companion()
    n, n_test_min = 40, 1
    grid = m._default_ncal_grid(n, n_test_min)
    assert grid and all(0 < g <= n - n_test_min for g in grid)


def test_companion_split_guard_rejects_oversized_ncal():
    """conformal_sweep skips n_cal > n_cases (the old min()-based guard let it slip through);
    an all-infeasible grid returns before any (core-lib) scoring, so this needs no core.
    """
    m = _load_companion()
    cases = [
        {"pred": np.zeros((10, 1)), "true": np.ones((10, 1)), "sigma": None}
        for _ in range(6)
    ]
    out = m.conformal_sweep(
        cases, alpha=0.1, arms=("constant",), n_cal_grid=(100,), n_test_min=1
    )
    assert out["arms"]["constant"] == {}


def _write_export_case(
    export_dir,
    name,
    *,
    field="pressure",
    n=32,
    dataset=None,
    with_std=True,
    seed=0,
):
    """Write one synthetic engine-export npz (optionally with the ``dataset`` label key)."""
    rng = np.random.default_rng(seed)
    arrays = {
        f"pred_{field}": rng.normal(0.0, 1.0, n),
        f"true_{field}": rng.normal(0.0, 1.0, n),
    }
    if with_std:
        arrays[f"std_{field}"] = np.abs(rng.normal(1.0, 0.1, n))
    if dataset is not None:
        arrays["dataset"] = np.array(dataset)
    np.savez(export_dir / f"{name}.npz", **arrays)


def test_companion_export_groups_by_dataset(tmp_path):
    """load_cases_from_export groups cases per the npz 'dataset' key; a missing key falls back
    to the 'unknown' label."""
    m = _load_companion()
    export = tmp_path / "conformal_inputs" / "modelA"
    export.mkdir(parents=True)
    _write_export_case(export, "dsA_case1", dataset="dsA", seed=1)
    _write_export_case(export, "dsA_case2", dataset="dsA", seed=2)
    _write_export_case(export, "dsB_case1", dataset="dsB", seed=3)
    _write_export_case(export, "legacy_case", dataset=None, seed=4)
    groups = m.load_cases_from_export(str(tmp_path), "modelA")
    assert set(groups) == {"dsA", "dsB", "unknown"}
    assert [len(groups[k]) for k in ("dsA", "dsB", "unknown")] == [2, 1, 1]
    case = groups["dsA"][0]
    assert case["pred"].shape == (32, 1) and case["true"].shape == (32, 1)
    assert case["sigma"].shape == (32, 1)


def test_companion_export_std_omission_gives_none_sigma(tmp_path):
    """A deterministic export (no std_<field>) loads with sigma=None (adaptive arm skipped)."""
    m = _load_companion()
    export = tmp_path / "conformal_inputs" / "modelA"
    export.mkdir(parents=True)
    _write_export_case(export, "dsA_case1", dataset="dsA", with_std=False)
    groups = m.load_cases_from_export(str(tmp_path), "modelA")
    assert set(groups) == {"dsA"}
    assert groups["dsA"][0]["sigma"] is None


def test_companion_export_field_aliasing(tmp_path):
    """Arrays exported under wall_shear_stress resolve through the shear_stress alias, keeping
    the (N, 3) vector components for the amax risk event."""
    m = _load_companion()
    export = tmp_path / "conformal_inputs" / "modelA"
    export.mkdir(parents=True)
    n = 16
    np.savez(
        export / "dsA_case1.npz",
        pred_wall_shear_stress=np.zeros((n, 3)),
        true_wall_shear_stress=np.ones((n, 3)),
        std_wall_shear_stress=np.ones((n, 3)),
        dataset=np.array("dsA"),
    )
    groups = m.load_cases_from_export(str(tmp_path), "modelA", field="shear_stress")
    assert set(groups) == {"dsA"}
    assert groups["dsA"][0]["pred"].shape == (n, 3)
    # No export for this field name at all -> empty dict, so the caller can fall back.
    assert m.load_cases_from_export(str(tmp_path), "no_such_model") == {}


def test_companion_sweep_runs_with_core_calibrator():
    """End-to-end conformal_sweep through the core RiskControlCalibrator (skipped when the
    core conformal branch is not on the path — the only genuinely core-dependent path).
    """
    pytest.importorskip("physicsnemo.experimental.uq.conformal")
    m = _load_companion()
    rng = np.random.default_rng(0)
    cases = [
        {
            "pred": rng.normal(0.0, 1.0, (50, 1)),
            "true": rng.normal(0.0, 1.0, (50, 1)),
            "sigma": None,
        }
        for _ in range(12)
    ]
    out = m.conformal_sweep(
        cases,
        alpha=0.2,
        arms=("constant",),
        n_cal_grid=(8,),
        n_test=4,
        n_splits=5,
    )
    res = out["arms"]["constant"][8]
    assert 0.0 <= res["coverage_med"] <= 1.0
    assert res["width_med"] > 0.0
    assert out["n_cases"] == 12


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
