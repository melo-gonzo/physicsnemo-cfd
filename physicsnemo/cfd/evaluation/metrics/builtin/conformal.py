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

"""Conformal *diagnostic* metric: approximate resplit coverage + width, as a benchmark column.

**This is a diagnostic, NOT a guaranteed calibration artifact.** It reports a repeated
random-resplit estimate of conformal coverage/width so a conformal signal sits in the same table
as the nominal UQ metrics (NLPD / coverage@95 / z-RMS / sharpness / AUSE). It is an approximate
*stability* diagnostic — the number is a median over ``n_splits`` random calibration/test resplits
of the *evaluation* cases, not a single fitted-then-held-out guarantee, and it is NOT a persisted
deployment artifact. The **library-exact reference** for the same risk *event* is the companion
conformal *report* (``conformal_analysis.py``), which reuses the core
``physicsnemo.experimental.uq.conformal`` calibrator end to end on the saved per-case fields — but
it too reports approximate-resplit coverage/width, not a single fitted-then-held-out guarantee. A
distribution-free *guarantee* awaits (b) an engine-integrated calibrator with explicit
train/calibration/test IDs that persists one deployment artifact. Do not cite this column as a
guarantee.

**Risk event (what the reported number controls).** For each surface *field* (pressure; wss) a
spatial POINT is declared *miscovered* when **any** requested vector component escapes the band —
the per-point nonconformity score is the max over the field's components, so a vector field yields
ONE risk per field, not one per component. This is the fixed point-event loss in the core
``RiskControlCalibrator`` (a true point fraction) and the companion ``conformal_analysis.py``;
both paths therefore control the *same* event. The loss weights points uniformly (point-count
weighting).

Per field and as a headline ``mean_*`` it reports (all in **physical units**):

* ``diag_coverage_const`` / ``diag_width_const``   — resplit band from the **residual** score
  (``AbsoluteErrorScore``); defined for *any* model, including the deterministic baseline ("baseline
  mode": no model σ used).
* ``diag_coverage_adaptive`` / ``diag_width_adaptive`` — resplit band from the **σ-normalized**
  score (the model's own ``std``); ``NaN`` for a model with no ``std``.

Risk is **point-count weighted** (uniform over points/cells). Area/volume weighting — the more
physically credible risk for CFD — is *planned in the core path* (it needs per-point weights
through the core calibrator) and is not yet available in either path; this lean diagnostic
therefore reports the point-count-weighted risk only.

**Method.** ``partial`` emits a compact per-case sorted-score sketch (upper empirical quantiles)
that is read *conservatively* via a lower-bound CDF mapping — miscoverage read off a sketch never
under-estimates the true empirical miscoverage (see ``_sketch`` / ``_loss_on_grid``); ``finalize``
does ``n_splits`` random cal/test resplits, and for each split builds the λ-candidate grid from the
**calibration split only** (no test leakage), fits λ̂ by ``n/(n+1)·R̂(λ)+1/(n+1) ≤ α``, and
measures coverage/width on the held-out cases. Infeasible splits (``α < 1/(n_cal+1)``) yield
``NaN`` rather than a silently-maxed λ. ``alpha`` (miscoverage target), ``n_splits`` (resplits),
``test_frac`` (held-out fraction per split), ``sketch_q`` (sketch resolution) and ``sigma_eps``
(σ floor) are all configurable per run via the metric spec exactly like ``alpha``
(``{name: conformal_diagnostic, alpha: 0.1, n_splits: 40, test_frac: 0.4, sketch_q: 129}``).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from physicsnemo.cfd.evaluation.datasets.schema import as_distribution
from physicsnemo.cfd.postprocessing_tools.metric_registry import register_metric

logger = logging.getLogger(__name__)

# Design note: this is a self-contained *diagnostic*, deliberately kept dependency-light so the
# table column runs without the core conformal branch installed. It is NOT the source of truth:
# the CFD engine should CONSUME the core ``physicsnemo.experimental.uq.conformal`` calibrator
# (the companion report already does) rather than duplicate CRC math for anything cited as a
# guarantee. This module survives only as an approximate, clearly-labelled stability diagnostic.

#: Default number of sorted-score quantiles retained per case (miscoverage-curve sketch resolution).
#: Overridable per run via the ``sketch_q`` metric-spec key.
_DEFAULT_SKETCH_Q = 129
#: Default σ floor for the normalized (adaptive) score. Single-source MIRROR of the core
#: ``physicsnemo.experimental.uq.conformal.NormalizedErrorScore(eps=1e-8)`` default, duplicated here
#: (not imported) so the diagnostic runs without the core conformal branch installed; if the core
#: default changes, update this mirror. Overridable per run via the ``sigma_eps`` metric-spec key.
_DEFAULT_SIG_EPS = 1e-8
#: Default repeated calibration/test splits averaged in finalize (median over draws).
#: Overridable per run via the ``n_splits`` metric-spec key.
_DEFAULT_N_SPLITS = 40
#: Default fraction of cases held out as the conformal test set per split.
#: Overridable per run via the ``test_frac`` metric-spec key.
_DEFAULT_TEST_FRAC = 0.4
#: Schema constant: component suffixes for 3-vector fields (matches the nominal UQ metrics). The
#: per-point risk event reduces over these components with ``amax`` (see the module docstring).
_VEC3 = ("x", "y", "z")


def _to_f64(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    detach = getattr(x, "detach", None)
    if callable(detach):
        x = detach()
    cpu = getattr(x, "cpu", None)
    if callable(cpu):
        x = cpu()
    return np.asarray(x, dtype=np.float64)


def _sketch(scores: np.ndarray, sketch_q: int = _DEFAULT_SKETCH_Q) -> np.ndarray:
    """Sorted-score sketch: the **upper** empirical quantile (``method='higher'`` → an actual
    data value, never interpolated below a score) at ``sketch_q`` evenly spaced probabilities.

    Guarantee (with the read in :func:`_loss_on_grid`): ``sketch[g]`` is a real score whose
    empirical CDF is ``>= g/(Q-1)`` (upper quantile), so the lower-bound CDF read — coverage
    certified as ``(m-1)/(Q-1)`` from ``m`` sketch values ``<= λ`` — never over-states coverage.
    Equivalently, miscoverage read off the sketch never under-estimates the true empirical
    miscoverage (conservative, matching the core sketch convention); it can over-estimate it by
    at most ``~1/(Q-1)`` plus quantile granularity.
    """
    return np.quantile(scores, np.linspace(0.0, 1.0, sketch_q), method="higher")


def _loss_on_grid(sketch: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Conservative per-case miscoverage read ``L(λ) >= P(score>λ)`` at each grid λ.

    With ``m`` sketch values ``<= λ``, the upper-quantile construction of :func:`_sketch`
    certifies an empirical CDF of at least ``(m-1)/(Q-1)`` at λ, so the returned loss
    ``1 - (m-1)/(Q-1)`` never under-estimates the true miscoverage. Below the sketch minimum
    (``m == 0``, i.e. λ smaller than every retained score) the loss is clamped to ``1.0``.
    """
    q = sketch.shape[0]
    m = np.searchsorted(sketch, grid, side="right")
    # m == 0 (λ below the sketch minimum) and m == 1 both certify nothing -> loss 1.0.
    return 1.0 - np.maximum(m - 1, 0) / max(q - 1, 1)


def _crc_lambda(calib_sketches: np.ndarray, grid: np.ndarray, alpha: float) -> float:
    """Risk-control λ̂ = inf{λ : n/(n+1)·R̂(λ) + 1/(n+1) ≤ α} over the calibration cases.

    ``grid`` must be built from the **calibration** scores only (the caller does this per split,
    so no held-out information enters the candidate set). Returns ``NaN`` when no candidate is
    feasible — never a silently-maxed λ.
    """
    n = calib_sketches.shape[0]
    mean_loss = np.mean([_loss_on_grid(s, grid) for s in calib_sketches], axis=0)
    bound = (n / (n + 1.0)) * mean_loss + 1.0 / (n + 1.0)
    ok = np.nonzero(bound <= alpha)[0]
    return float(grid[ok[0]]) if ok.size else float("nan")


def _iter_fields(gt: dict, predictions: dict):
    """Yield ``(field, resid, sigma_or_None)`` in physical units, per FIELD (not per component).

    ``resid`` is the ``(N, C)`` absolute residual and ``sigma`` is ``(N, C)`` (or ``None``); the
    per-point risk event (a point fails if ANY of its ``C`` components escapes the band) is applied
    by the caller as an ``amax`` over the ``C`` columns, mirroring the core
    ``RiskControlCalibrator``'s fixed trailing-component reduction. For 3-vector fields the ``C``
    columns are the :data:`_VEC3` components.

    Non-finite guard (skip-and-warn, matching the sibling UQ metrics): a case-field with any
    non-finite ground truth or mean is **skipped** with a warning — a NaN sketch on the test side
    would otherwise count as fully covered (``NaN > λ`` is ``False``) and *inflate* reported
    coverage. A non-finite or negative ``std`` drops only the adaptive (σ-normalized) arm for
    that case-field.
    """
    if not gt:
        return
    for key in predictions:
        if key not in gt or gt[key] is None:
            continue
        dist = as_distribution(predictions, key)
        if dist is None:
            continue
        y, mu = _to_f64(gt[key]), _to_f64(dist.mean)
        sig = _to_f64(dist.std) if dist.std is not None else None
        if y is None or mu is None:
            continue

        def _2d(a):
            if a is None:
                return None
            return a.reshape(-1, 1) if a.ndim == 1 else (a if a.ndim == 2 else None)

        y2, mu2, sig2 = _2d(y), _2d(mu), _2d(sig)
        if y2 is None or mu2 is None or y2.shape != mu2.shape:
            continue
        if sig2 is not None and sig2.shape != y2.shape:
            sig2 = None
        if not (np.all(np.isfinite(y2)) and np.all(np.isfinite(mu2))):
            logger.warning(
                "Conformal diagnostic: field %r has non-finite ground truth or mean "
                "for this case; skipping the case-field so it cannot inflate "
                "reported coverage.",
                key,
            )
            continue
        if sig2 is not None and (not np.all(np.isfinite(sig2)) or np.any(sig2 < 0.0)):
            logger.warning(
                "Conformal diagnostic: field %r has non-finite or negative std for "
                "this case; dropping the adaptive (σ-normalized) arm for this case.",
                key,
            )
            sig2 = None
        yield key, np.abs(y2 - mu2), sig2


class _ConformalDiagnostic:
    """Approximate conformal coverage/width *diagnostic* (constant + adaptive arms), point-weighted.

    Sample metric: ``partial`` emits a per-case conservative score sketch; ``finalize`` runs the
    repeated-resplit CRC diagnostic. NOT a guaranteed artifact — see the module docstring. The
    per-run knobs (``alpha``, ``n_splits``, ``test_frac``, ``sketch_q``, ``sigma_eps``) are the
    registration defaults; each may be overridden per run through the metric spec.
    """

    def __init__(
        self,
        alpha: float,
        *,
        n_splits: int = _DEFAULT_N_SPLITS,
        test_frac: float = _DEFAULT_TEST_FRAC,
        sketch_q: int = _DEFAULT_SKETCH_Q,
        sigma_eps: float = _DEFAULT_SIG_EPS,
    ) -> None:
        self.alpha = float(alpha)
        self.n_splits = int(n_splits)
        self.test_frac = float(test_frac)
        self.sketch_q = int(sketch_q)
        self.sigma_eps = float(sigma_eps)

    # -- phase 1: one compact summary per case -------------------------------------------------
    def partial(
        self,
        gt: Any,
        predictions: Any,
        *,
        alpha: float | None = None,
        n_splits: int | None = None,
        test_frac: float | None = None,
        sketch_q: int | None = None,
        sigma_eps: float | None = None,
        **_: Any,
    ) -> dict[str, float]:
        a = float(alpha if alpha is not None else self.alpha)
        q = int(sketch_q if sketch_q is not None else self.sketch_q)
        ns = int(n_splits if n_splits is not None else self.n_splits)
        tf = float(test_frac if test_frac is not None else self.test_frac)
        eps = float(sigma_eps if sigma_eps is not None else self.sigma_eps)
        # Stash the resolved per-run knobs so finalize_samples (which sees only the collected
        # per-case columns) reconstructs the sketch and resplits with the identical settings.
        stats: dict[str, float] = {
            "_alpha": a,
            "_sketch_q": float(q),
            "_n_splits": float(ns),
            "_test_frac": tf,
        }
        for field, resid, sig in _iter_fields(gt or {}, predictions or {}):
            # amax over components: a point is miscovered if ANY component escapes the band.
            const_score = resid.max(axis=1)
            csk = _sketch(const_score, q)
            for g in range(q):
                stats[f"{field}::c{g}"] = float(csk[g])
            if sig is not None:
                s = np.maximum(sig, eps)
                # Normalize per component, THEN reduce with amax (the reduction must follow the
                # σ-normalization, matching the core's fixed point-event reduction).
                adapt_score = (resid / s).max(axis=1)
                ask = _sketch(adapt_score, q)
                for g in range(q):
                    stats[f"{field}::a{g}"] = float(ask[g])
                stats[f"{field}::sig"] = float(np.mean(s))
        return stats

    # -- phase 2: resplit-calibrate on cal, measure on the disjoint test -----------------------
    def finalize_samples(self, collected: dict[str, list[float]]) -> dict[str, float]:
        def _first(key: str, default: float) -> float:
            vals = collected.get(key)
            return float(vals[0]) if vals else float(default)

        alpha = _first("_alpha", self.alpha)
        sketch_q = int(_first("_sketch_q", self.sketch_q))
        n_splits = int(_first("_n_splits", self.n_splits))
        test_frac = _first("_test_frac", self.test_frac)
        # Regroup "{field}::{stat}" -> per field.
        by_field: dict[str, dict[str, list[float]]] = {}
        for key, vals in collected.items():
            if "::" not in key:
                continue
            field, stat = key.rsplit("::", 1)
            by_field.setdefault(field, {})[stat] = vals
        if not by_field:
            return self._empty()

        rng = np.random.default_rng(0)
        out: dict[str, float] = {}
        agg = {k: [] for k in ("cov_const", "width_const", "cov_adapt", "width_adapt")}
        for field in sorted(by_field):
            d = by_field[field]
            const = self._stack(d, "c", sketch_q)
            if const is None:
                continue
            res = self._eval_arm(
                const,
                sig=None,
                rng=rng,
                alpha=alpha,
                n_splits=n_splits,
                test_frac=test_frac,
            )
            out[f"{field}_diag_coverage_const"] = res[0]
            out[f"{field}_diag_width_const"] = res[1]
            agg["cov_const"].append(res[0])
            agg["width_const"].append(res[1])

            adapt = self._stack(d, "a", sketch_q)
            sig = np.asarray(d.get("sig", []), dtype=np.float64)
            if adapt is not None and sig.size == adapt.shape[0]:
                res = self._eval_arm(
                    adapt,
                    sig=sig,
                    rng=rng,
                    alpha=alpha,
                    n_splits=n_splits,
                    test_frac=test_frac,
                )
                out[f"{field}_diag_coverage_adaptive"] = res[0]
                out[f"{field}_diag_width_adaptive"] = res[1]
                agg["cov_adapt"].append(res[0])
                agg["width_adapt"].append(res[1])

        out["mean_diag_coverage_const"] = _nanmean(agg["cov_const"])
        out["mean_diag_width_const"] = _nanmean(agg["width_const"])
        out["mean_diag_coverage_adaptive"] = _nanmean(agg["cov_adapt"])
        out["mean_diag_width_adaptive"] = _nanmean(agg["width_adapt"])
        out["diag_alpha"] = float(alpha)
        return out

    # -- helpers -------------------------------------------------------------------------------
    def _stack(
        self, d: dict[str, list[float]], prefix: str, sketch_q: int
    ) -> np.ndarray | None:
        """Reconstruct (n_cases, Q) sketches from the per-quantile columns."""
        cols = [d.get(f"{prefix}{g}") for g in range(sketch_q)]
        if any(c is None for c in cols):
            return None
        arr = np.array(cols, dtype=np.float64).T  # (n_cases, Q)
        arr.sort(axis=1)  # sketches are already sorted, but guard against fp noise
        return arr if arr.shape[0] >= 4 else None

    def _eval_arm(
        self,
        sketches: np.ndarray,
        sig,
        rng,
        alpha: float,
        *,
        n_splits: int = _DEFAULT_N_SPLITS,
        test_frac: float = _DEFAULT_TEST_FRAC,
    ) -> tuple[float, float]:
        """Median resplit coverage + physical width over repeated calib/test splits.

        Leak-free: each split builds its λ-candidate grid from the **calibration** scores only.
        Infeasible splits (``α < 1/(n_cal+1)``) contribute nothing; all-infeasible → ``NaN``.
        """
        n = sketches.shape[0]
        n_test = max(2, round(test_frac * n))
        n_cal = n - n_test
        if n_cal < 2 or alpha < 1.0 / (n_cal + 1.0):
            return float("nan"), float("nan")
        covs, widths = [], []
        for _ in range(n_splits):
            idx = rng.permutation(n)
            cal, test = idx[:n_cal], idx[n_cal:]
            # Candidate λ grid from CALIBRATION scores only (conservative, bounded resolution) —
            # no held-out information enters the candidate set.
            grid = np.unique(
                np.quantile(
                    sketches[cal].reshape(-1),
                    np.linspace(0.0, 0.9995, 2048),
                    method="higher",
                )
            )
            if grid.size == 0:
                continue
            lam = _crc_lambda(sketches[cal], grid, alpha)
            if not np.isfinite(lam):
                continue  # infeasible on this split
            # Miscoverage of each held-out case at λ, via the same conservative lower-bound
            # CDF read as calibration — reported coverage never over-states the truth.
            lam_grid = np.asarray([lam])
            risk = float(
                np.mean([float(_loss_on_grid(sketches[t], lam_grid)[0]) for t in test])
            )
            covs.append(1.0 - risk)
            widths.append(
                2.0 * lam if sig is None else 2.0 * lam * float(np.mean(sig[test]))
            )
        if not covs:
            return float("nan"), float("nan")
        return float(np.median(covs)), float(np.median(widths))

    def _empty(self) -> dict[str, float]:
        nan = float("nan")
        return {
            "mean_diag_coverage_const": nan,
            "mean_diag_width_const": nan,
            "mean_diag_coverage_adaptive": nan,
            "mean_diag_width_adaptive": nan,
        }


def _nanmean(xs: list[float]) -> float:
    xs = [x for x in xs if not np.isnan(x)]
    return float(np.mean(xs)) if xs else float("nan")


def register_conformal_metrics(
    alpha: float = 0.1,
    *,
    n_splits: int = _DEFAULT_N_SPLITS,
    test_frac: float = _DEFAULT_TEST_FRAC,
    sketch_q: int = _DEFAULT_SKETCH_Q,
    sigma_eps: float = _DEFAULT_SIG_EPS,
) -> None:
    """Register the conformal *diagnostic* metric for both domains.

    ``alpha`` is the diagnostic miscoverage target (nominal coverage = 1-alpha); defaults to 0.1
    (90%). It, along with ``n_splits`` (resplits), ``test_frac`` (held-out fraction), ``sketch_q``
    (sketch resolution) and ``sigma_eps`` (σ floor), sets the registration default and is
    overridable per run via the metric spec (``{name: conformal_diagnostic, alpha: ...,
    n_splits: ..., test_frac: ..., sketch_q: ...}``). Registered under ``conformal_diagnostic``
    (and the legacy alias ``conformal_crc``); the name ``_diagnostic`` is deliberate — this is not
    a guaranteed calibration artifact.
    """
    metric = _ConformalDiagnostic(
        alpha=alpha,
        n_splits=n_splits,
        test_frac=test_frac,
        sketch_q=sketch_q,
        sigma_eps=sigma_eps,
    )
    for domain in ("surface", "volume"):
        register_metric("conformal_diagnostic", metric, domain=domain)
        register_metric("conformal_crc", metric, domain=domain)  # legacy alias
