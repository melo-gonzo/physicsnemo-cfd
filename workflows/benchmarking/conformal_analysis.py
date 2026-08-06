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

"""Conformal analysis over a UQ-benchmark run — the library-backed companion.

Run *after* the benchmark. It reads the per-case surface fields the benchmark writes and runs our
conformal library (``physicsnemo.experimental.uq.conformal``) over the cases to answer "what does
the distribution-free machinery add, and how does it compare across models?":

* **Diagnostic (approximate resplit) coverage + width** (constant / residual = *baseline mode*,
  and adaptive / σ-normalized), the exact counterpart of the in-table ``conformal_diagnostic``
  metric.
* The **holdout-size sweep** (coverage & width vs number of calibration cases) — the
  finite-sample story the single table number cannot show.

**Risk event (unified with the metric).** Per surface *field* a spatial POINT is miscovered when
**any** requested vector component escapes the band: the per-point nonconformity score is reduced
over the field's components with ``amax`` before thresholding, exactly as the core
``RiskControlCalibrator(channel_reduction="amax")`` (its default) and the in-table
``conformal_diagnostic`` metric do. A vector field (WSS) therefore yields ONE risk per field, not
one lambda pooled over raveled components. λ̂ itself is fit by the core ``RiskControlCalibrator``
end to end, so this is the library-exact reference for the *event*; the reported coverage/width
are still an approximate resplit view (repeated random cal/test splits of the evaluation cases),
not a single fitted-then-held-out guarantee.

**Inputs.** Prefers the engine's compact per-case export
``<results_dir>/conformal_inputs/<model>/<dataset_label>_<case>.npz`` (``pred_<field>`` /
``true_<field>`` / ``std_<field>`` arrays in physical units plus a ``dataset`` string key naming
the case's dataset; ``run.conformal_export: true``). Cases are **grouped per dataset** (missing
``dataset`` key → label ``"unknown"``) and the sweep runs per ``(model, dataset)`` group —
calibration scores are exchangeable within a dataset, not across datasets, so pooling would mix
score distributions. An optional ``--datasets`` filter restricts the analysis to named labels.
Falls back to the per-case inference meshes (``inference_<model>_<dataset>_<case>.vtp`` with
``Pred*`` / ``Std*`` / ``True*`` arrays; ``run.save_inference_mesh: true``) when the export
directory is absent — the meshes carry no dataset labels, so the fallback pools ALL cases under
``dataset="pooled(vtp-fallback)"`` with a loud warning.

Writes ``conformal_analysis.json`` (``{model: {dataset: {..., "n_cases": ...}}}``) +
``conformal_sweep_<field>.png`` next to the benchmark results.

Env: ``physicsnemo-cfd`` on the path always; the core conformal library
(``physicsnemo.experimental.uq.conformal``, currently on the ``conformal-uq`` line of
physicsnemo-core — set ``PYTHONPATH`` accordingly) is imported **lazily**, only by the scoring /
calibration steps, so the loading, grouping and plotting helpers work without it.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from typing import Any

import numpy as np
import torch

# NOTE: the core conformal library (physicsnemo.experimental.uq.conformal) is imported LAZILY
# inside _reduced_score / _lam_hat so the pure-logic pieces (split guards, export loading and
# grouping, plotting) import and run without the core branch installed.

logger = logging.getLogger(__name__)

# Channel order of the standard surface fields (matches the wrappers / nominal metrics).
_VEC3 = ("x", "y", "z")
#: Default σ floor for the adaptive (σ-normalized) arm; mirrors the core NormalizedError default.
_SIGMA_EPS = 1e-8


# --------------------------------------------------------------------------- data model
def _reduced_score(
    pred: np.ndarray,
    true: np.ndarray,
    sigma: np.ndarray | None,
    arm: str,
    eps: float = _SIGMA_EPS,
) -> np.ndarray:
    """Per-POINT nonconformity score for one case/arm, via the library score objects.

    Components are reduced with ``amax`` *after* scoring (``NormalizedError`` normalizes each
    component by its own σ first), so a point is miscovered when ANY component escapes the band —
    the exact loss ``RiskControlCalibrator(channel_reduction="amax")`` certifies. Imports the
    core conformal library lazily (only callers that actually score need it on the path).
    """
    from physicsnemo.experimental.uq.conformal import AbsoluteError, NormalizedError

    p = torch.from_numpy(np.ascontiguousarray(pred, dtype=np.float64))
    t = torch.from_numpy(np.ascontiguousarray(true, dtype=np.float64))
    if arm == "constant":
        s = AbsoluteError().score(p, t).numpy()
    else:
        sig = torch.from_numpy(np.maximum(sigma, eps).astype(np.float64))
        s = NormalizedError(eps=eps).score(p, t, aux={"sigma": sig}).numpy()
    if s.ndim > 1:
        s = s.max(axis=tuple(range(1, s.ndim)))
    return s.ravel()


def _lam_hat(cal_scores: list[np.ndarray], alpha: float) -> float:
    """Exact risk-control λ̂ from the core ``RiskControlCalibrator`` over the calibration cases.

    ``cal_scores`` are already the per-point (component-reduced) nonconformity vectors, one per
    calibration case; each is presented as one sample (``pred=0``, ``target=score``) so the
    calibrator recomputes ``|score|`` unchanged and fits the CRC lambda over the unified event.
    Imports the core conformal library lazily.
    """
    from physicsnemo.experimental.uq.conformal import (
        AbsoluteError,
        RiskControlCalibrator,
    )

    calib = RiskControlCalibrator(AbsoluteError(), alpha=alpha)
    for s in cal_scores:
        n = s.shape[0]
        calib.update(
            torch.zeros(n, dtype=torch.float64),
            torch.from_numpy(s.astype(np.float64)),
        )
    lam = calib.finalize(distributed=False).lam
    return float(lam if not isinstance(lam, dict) else next(iter(lam.values())))


def _default_ncal_grid(n: int, n_test_min: int) -> tuple[int, ...]:
    """A calibration-size grid whose entries are all ``<= n - n_test_min`` (leave a held-out set)."""
    cap = n - n_test_min
    grid = tuple(g for g in (8, 16, 32, 64, 128, 256) if 2 <= g <= cap)
    if not grid and cap >= 2:
        grid = (cap,)
    return grid


def conformal_sweep(
    cases: list[dict[str, np.ndarray]],
    *,
    alpha: float = 0.1,
    arms: tuple[str, ...] = ("constant", "adaptive"),
    n_cal_grid: tuple[int, ...] | None = None,
    n_test: int = 100,
    n_test_min: int = 1,
    n_splits: int = 40,
    sigma_eps: float = _SIGMA_EPS,
    seed: int = 0,
) -> dict[str, Any]:
    """Library-exact diagnostic (approximate resplit) coverage + width vs n_cal, per arm, one field.

    ``cases`` is a list of per-case dicts with keys ``pred`` / ``true`` (``(N, C)``, physical
    units) and ``sigma`` (``(N, C)`` or ``None``: deterministic → adaptive arm skipped), and
    should all belong to ONE dataset group — the caller (``main``) runs one sweep per
    ``(model, dataset)`` since calibration scores are exchangeable within a dataset, not across.
    Repeated disjoint calib/test splits give a median + 5–95 band, matching the metric
    methodology. When ``n_cal_grid`` is ``None`` it defaults to values
    ``<= len(cases) - n_test_min``.
    """
    rng = np.random.default_rng(seed)
    n = len(cases)
    has_sigma = all(c.get("sigma") is not None for c in cases)
    active_arms = tuple(a for a in arms if not (a == "adaptive" and not has_sigma))

    grid = n_cal_grid if n_cal_grid is not None else _default_ncal_grid(n, n_test_min)
    # Feasibility filter BEFORE scoring. Non-empty held-out set (P1-4): 0 < n_cal <= n -
    # n_test_min, written as an explicit upper bound so it is NOT bypassed when n_cal > n (the
    # old min()-based guard let a negative test size slip through). Plus CRC feasibility
    # (n_cal >= 2, alpha >= 1/(n_cal+1)). Filtering first also means an all-infeasible grid
    # returns without touching the (lazily imported) core scoring path.
    feasible = [
        int(n_cal)
        for n_cal in grid
        if 0 < n_cal <= n - n_test_min and n_cal >= 2 and alpha >= 1.0 / (n_cal + 1)
    ]
    out: dict[str, Any] = {"alpha": alpha, "n_cases": n, "arms": {}}
    for a in active_arms:
        out["arms"][a] = {}
    if not feasible:
        return out

    # Pre-score every case per arm once (the expensive part), and cache each case's mean sigma.
    scored = {a: [] for a in active_arms}
    sig_mean = np.array(
        [
            float(np.mean(np.maximum(c["sigma"], sigma_eps))) if has_sigma else 1.0
            for c in cases
        ]
    )
    for c in cases:
        for a in active_arms:
            scored[a].append(
                _reduced_score(c["pred"], c["true"], c.get("sigma"), a, sigma_eps)
            )

    for a in active_arms:
        per_ncal = out["arms"][a]
        for n_cal in feasible:
            nt = min(n_test, n - n_cal)
            covs, widths = [], []
            for _ in range(n_splits):
                idx = rng.permutation(n)
                cal, test = idx[:n_cal], idx[n_cal : n_cal + nt]
                lam = _lam_hat([scored[a][i] for i in cal], alpha)
                risk = np.mean([np.mean(scored[a][t] > lam) for t in test])
                covs.append(1.0 - float(risk))
                widths.append(
                    2.0
                    * lam
                    * (1.0 if a == "constant" else float(np.mean(sig_mean[test])))
                )
            per_ncal[int(n_cal)] = {
                "coverage_med": float(np.median(covs)),
                "coverage_lo": float(np.percentile(covs, 5)),
                "coverage_hi": float(np.percentile(covs, 95)),
                "width_med": float(np.median(widths)),
            }
    return out


# --------------------------------------------------------------------------- I/O helpers
def _as2d(a: Any) -> np.ndarray:
    """View an ``(N,)`` scalar field or ``(N, C)`` vector field as ``(N, C)`` (keeps components)."""
    arr = np.asarray(a, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


# The predictions-only inference mesh the benchmark writes carries: the wrapper predictions
# (``<field>_pred`` / ``<field>_std`` / ``<field>_epistemic_std``, per ``output.*_mesh_field_names``)
# AND the prepared source arrays, in which the *ground truth* lives under the DrivAerML-convention
# names the adapter renamed to (``pMeanTrim`` for pressure, ``wallShearStressMeanTrim`` for WSS) —
# NOT ``<field>_true`` (that name only appears on the separate comparison mesh). So resolve truth
# from those source names.
_FIELD_ARRAYS: dict[str, dict[str, tuple[str, ...]]] = {
    "pressure": {
        "pred": ("pressure_pred",),
        "std": ("pressure_std",),
        "true": ("pMeanTrim", "pressure_true", "Pressure"),
    },
    "shear_stress": {
        "pred": ("wall_shear_stress_pred",),
        "std": ("wall_shear_stress_std",),
        "true": ("wallShearStressMeanTrim", "wall_shear_stress_true"),
    },
    "wall_shear_stress": {
        "pred": ("wall_shear_stress_pred",),
        "std": ("wall_shear_stress_std",),
        "true": ("wallShearStressMeanTrim", "wall_shear_stress_true"),
    },
}

#: Field-name aliases used to resolve ``<role>_<field>`` keys in the conformal_inputs export.
_FIELD_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "pressure": ("pressure",),
    "shear_stress": ("wall_shear_stress", "shear_stress"),
    "wall_shear_stress": ("wall_shear_stress", "shear_stress"),
}


def load_cases_from_export(
    results_dir: str, model: str, field: str = "pressure"
) -> dict[str, list[dict]]:
    """Read per-case ``(pred, true, sigma)`` for one model/field from the engine's export,
    **grouped per dataset**.

    Reads ``<results_dir>/conformal_inputs/<model>/<dataset_label>_<case>.npz``
    (``pred_<field>`` / ``true_<field>`` / ``std_<field>`` in physical units; ``std_*`` omitted
    for a deterministic method). Each file's ``dataset`` key (a string array the engine writes)
    labels the case's dataset; files missing the key fall back to the label ``"unknown"``.
    Returns ``{dataset_label: [case, ...]}``; empty ``{}`` when the export directory is
    absent/empty, so the caller can fall back to meshes.
    """
    export_dir = os.path.join(results_dir, "conformal_inputs", model)
    if not os.path.isdir(export_dir):
        return {}
    names = _FIELD_NAME_ALIASES.get(field.lower(), (field.lower(),))

    def _pick(npz, role: str):
        for fname in names:
            key = f"{role}_{fname}"
            if key in npz.files:
                return npz[key]
        return None

    groups: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(export_dir, "*.npz"))):
        with np.load(path) as npz:
            pred, true = _pick(npz, "pred"), _pick(npz, "true")
            if pred is None or true is None:
                continue
            std = _pick(npz, "std")
            dataset = (
                str(np.asarray(npz["dataset"]).reshape(-1)[0])
                if "dataset" in npz.files
                else "unknown"
            )
            pred2d, true2d = _as2d(pred), _as2d(true)
            if pred2d.shape != true2d.shape:
                continue
            std2d = _as2d(std) if std is not None else None
            if std2d is not None and std2d.shape != pred2d.shape:
                std2d = None
        groups.setdefault(dataset, []).append(
            {"pred": pred2d, "true": true2d, "sigma": std2d}
        )
    return groups


def load_cases_from_meshes(
    results_dir: str, model: str, field: str = "pressure"
) -> list[dict]:
    """Read per-case ``(pred, true, sigma)`` for one model/field from the benchmark inference meshes.

    Looks for ``inference_<model>_*_<case>.vtp`` and reads the ``<field>_pred`` / ``<field>_std``
    prediction arrays plus the ground truth (``pMeanTrim`` / ``wallShearStressMeanTrim`` — the
    adapter's DrivAerML names). Vector fields keep their ``(N, C)`` components (the amax risk event
    reduces them per point). Returns [] if none found. Requires pyvista.

    The meshes carry NO dataset labels, so this fallback cannot separate multi-dataset runs —
    :func:`load_cases` pools its output under ``dataset="pooled(vtp-fallback)"`` and warns.
    """
    import pyvista as pv

    aliases = _FIELD_ARRAYS.get(field.lower())

    def _find(data: dict, role: str):
        """Resolve (field, role) to a mesh array: exact known name first (case-insensitive),
        then a generic ``<field>_<role>`` heuristic across conventions."""
        lower = {k.lower(): k for k in data}
        if aliases:
            for cand in aliases.get(role, ()):
                if cand.lower() in lower:
                    return lower[cand.lower()]
        f, r = field.lower(), role.lower()
        cands = {f"{f}_{r}", f"{r}{f}", f"{r}_{f}", f"{f}{r}"}
        for kl, k in lower.items():
            if kl in cands or (f in kl and r in kl):
                return k
        return None

    cases = []
    pat = os.path.join(results_dir, f"inference_{model}_*.vt[pu]")
    for path in sorted(glob.glob(pat)):
        m = pv.read(path)
        data = {**dict(m.cell_data), **dict(m.point_data)}
        pred_k = _find(data, "pred")
        true_k = _find(data, "true")
        std_k = _find(data, "std")
        if pred_k is None or true_k is None:
            continue
        pred2d, true2d = _as2d(data[pred_k]), _as2d(data[true_k])
        if pred2d.shape != true2d.shape:
            continue
        std2d = _as2d(data[std_k]) if std_k is not None else None
        if std2d is not None and std2d.shape != pred2d.shape:
            std2d = None
        cases.append({"pred": pred2d, "true": true2d, "sigma": std2d})
    return cases


def load_cases(
    results_dir: str, model: str, field: str = "pressure"
) -> tuple[dict[str, list[dict]], str]:
    """Per-case fields for one model/field, preferring the export dir over inference meshes.

    Returns ``(groups, source_label)`` with ``groups = {dataset_label: [case, ...]}``; empty
    ``{}`` when neither source is present. The mesh fallback carries no dataset labels, so its
    cases are pooled under the single label ``"pooled(vtp-fallback)"`` (with a loud warning that
    multi-dataset separation is unavailable on that path).
    """
    groups = load_cases_from_export(results_dir, model, field)
    if groups:
        return groups, "conformal_inputs export"
    cases = load_cases_from_meshes(results_dir, model, field)
    if not cases:
        return {}, "inference meshes"
    logger.warning(
        "conformal_analysis: no conformal_inputs export for model %r — falling back to "
        "inference meshes, which carry NO dataset labels. Multi-dataset separation is "
        "UNAVAILABLE on this path: all %d cases are pooled under "
        "dataset='pooled(vtp-fallback)'. Re-run the benchmark with run.conformal_export: "
        "true for per-dataset analysis.",
        model,
        len(cases),
    )
    return {"pooled(vtp-fallback)": cases}, "inference meshes"


# --------------------------------------------------------------------------- plotting
def plot_sweep(
    results_by_model: dict[str, dict[str, dict]], field: str, out_png: str
) -> None:
    """Coverage/width vs n_cal, one line per ``(model, dataset, arm)``.

    ``results_by_model`` is the nested ``{model: {dataset: sweep_result}}`` structure ``main``
    assembles (one ``conformal_sweep`` output per model/dataset group).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axc, axw) = plt.subplots(1, 2, figsize=(9, 3.8))
    for model, by_dataset in results_by_model.items():
        for dataset, res in by_dataset.items():
            for a, style in (("constant", "-"), ("adaptive", "--")):
                arm = res.get("arms", {}).get(a)
                if not arm:
                    continue
                ns = sorted(int(k) for k in arm)
                label = f"{model} · {dataset} · {a}"
                axc.plot(
                    ns,
                    [
                        (
                            arm[str(n)]["coverage_med"]
                            if str(n) in arm
                            else arm[n]["coverage_med"]
                        )
                        for n in ns
                    ],
                    style,
                    marker="o",
                    ms=3,
                    label=label,
                )
                axw.plot(
                    ns,
                    [
                        (
                            arm[str(n)]["width_med"]
                            if str(n) in arm
                            else arm[n]["width_med"]
                        )
                        for n in ns
                    ],
                    style,
                    marker="o",
                    ms=3,
                    label=label,
                )
    first = next(iter(next(iter(results_by_model.values())).values()))
    axc.axhline(1 - first["alpha"], color="k", ls=":", lw=1)
    axc.set_xscale("log")
    axc.set_xlabel("calibration cases $n_{cal}$")
    axc.set_ylabel("diagnostic (approximate resplit) coverage")
    axw.set_xscale("log")
    axw.set_xlabel("calibration cases $n_{cal}$")
    axw.set_ylabel("mean band width [phys]")
    axc.set_title(f"conformal coverage — {field}")
    axw.set_title(f"conformal width — {field}")
    axc.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-dir",
        required=True,
        help="benchmark output_dir with conformal_inputs/ (preferred) or inference_*.vtp",
    )
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--field", default="pressure")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="restrict the analysis to these dataset labels (default: all labels found; "
        "the vtp fallback only offers 'pooled(vtp-fallback)')",
    )
    args = ap.parse_args()

    results_by_model: dict[str, dict[str, dict]] = {}
    for model in args.models:
        groups, source = load_cases(args.results_dir, model, args.field)
        if args.datasets is not None:
            groups = {k: v for k, v in groups.items() if k in args.datasets}
        if not groups:
            print(f"[skip] no conformal inputs for {model}")
            continue
        results_by_model[model] = {}
        for dataset in sorted(groups):
            cases = groups[dataset]
            results_by_model[model][dataset] = conformal_sweep(cases, alpha=args.alpha)
            print(f"[{model} / {dataset}] {len(cases)} cases analyzed from {source}")
    with open(os.path.join(args.results_dir, "conformal_analysis.json"), "w") as f:
        json.dump(results_by_model, f, indent=2)
    if results_by_model:
        plot_sweep(
            results_by_model,
            args.field,
            os.path.join(args.results_dir, f"conformal_sweep_{args.field}.png"),
        )
    print("conformal analysis written to", args.results_dir)


if __name__ == "__main__":
    main()
