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

"""Canonical CFD case schema for dataset adapters and model wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

#: Array payloads on :class:`FieldDistribution` may be NumPy arrays or (for the on-device
#: metric path) framework tensors such as ``torch.Tensor``. We
#: intentionally do not import ``torch`` here to keep the ``datasets`` layer dependency-light;
#: metrics convert to NumPy at their boundary.
ArrayLike = Any

# Surface vs volume inference (which manifold the case uses). Combined surface+volume is deferred.
InferenceDomain = Literal["surface", "volume"]

_VALID_INFERENCE_DOMAINS = frozenset({"surface", "volume"})


def normalize_inference_domain_str(
    value: str, *, parameter: str = "inference_domain"
) -> InferenceDomain:
    """Return ``surface`` or ``volume`` after strip/lowercase; raise on typos."""
    normalized = value.strip().lower()
    if normalized not in _VALID_INFERENCE_DOMAINS:
        raise ValueError(f"{parameter} must be 'surface' or 'volume'; got {value!r}")
    return normalized  # type: ignore[return-value]


def coerce_inference_domain_or_default(
    raw: Any,
    *,
    default: InferenceDomain,
    parameter: str,
) -> InferenceDomain:
    """Treat ``None`` as *default*; otherwise validate strings (reject ``None`` sentinel typos separately)."""
    if raw is None:
        return default
    if isinstance(raw, str):
        return normalize_inference_domain_str(raw, parameter=parameter)
    raise ValueError(f"{parameter} must be 'surface', 'volume', or null (got {raw!r})")


@dataclass
class CanonicalCase:
    """Canonical representation of a single CFD case.

    ``mesh_path`` is the primary mesh: surface ``.vtp`` when ``inference_domain`` is
    ``surface``, or volume ``.vtu`` when ``inference_domain`` is ``volume``.
    Decode-time model outputs for metrics should follow :func:`build_predictions_dict` keys/shape.
    """

    case_id: str
    mesh_path: str
    mesh_type: str  # "point" | "cell" — field dof; "unknown" when GT extractor did not set location
    ground_truth: dict[str, Any] | None = (
        None  # surface: pressure, shear_stress; volume: pressure, velocity, …
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    inference_domain: InferenceDomain = "surface"
    #: Optional mesh already loaded by the dataset adapter (e.g. :class:`pyvista.PolyData` for
    #: surface, :class:`pyvista.UnstructuredGrid` for volume). When set, benchmarks and wrappers
    #: may skip a redundant ``pv.read(case.mesh_path)`` for the same case. Adapters are not
    #: required to populate this field.
    reference_geometry: Any | None = None
    #: Optional in-memory geometry mesh (:class:`pyvista.PolyData`) for the wrappers' SDF /
    #: geometry-embedding branch, replacing an on-disk STL. When set, the datapipe I/O derives
    #: ``stl_coordinates`` / ``stl_faces`` / ``stl_centers`` from it instead of globbing an STL
    #: file next to ``mesh_path``. Adapters where the surface *is* the geometry (e.g. DrivAerStar)
    #: can set this to avoid materializing an STL; adapters with a distinct geometry file leave it
    #: ``None`` to keep the file-based lookup.
    geometry: Any | None = None

    def __post_init__(self) -> None:
        if self.mesh_type not in ("point", "cell", "unknown"):
            raise ValueError("mesh_type must be 'point', 'cell', or 'unknown'")
        if self.inference_domain not in ("surface", "volume"):
            raise ValueError("inference_domain must be 'surface' or 'volume'")

    def get_ground_truth(self, key: str, default: Any = None) -> Any:
        """Return ground truth field by key, or default if missing."""
        if self.ground_truth is None:
            return default
        return self.ground_truth.get(key, default)


def build_predictions_dict(
    *,
    pressure: np.ndarray | None = None,
    shear_stress: np.ndarray | None = None,
    velocity: np.ndarray | None = None,
    turbulent_viscosity: np.ndarray | None = None,
    **extra: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build canonical ``decode_outputs`` payloads: numpy arrays as ``float32`` under canonical keys.

    **Use this helper everywhere** in wrapper :meth:`~physicsnemo.cfd.evaluation.models.model_registry.CFDModel.decode_outputs`
    (prefer keyword arguments over ad-hoc ``{...}`` dicts) so dtypes and keys stay consistent.

    Surface wrappers typically pass ``pressure`` and ``shear_stress``; volume wrappers add
    ``velocity`` and ``turbulent_viscosity``. Omit any argument or pass ``None`` to skip keys.
    Extra keyword arrays are normalized the same way (e.g. custom fields).
    """
    out: dict[str, np.ndarray] = {}
    if pressure is not None:
        out["pressure"] = np.asarray(pressure, dtype=np.float32)
    if shear_stress is not None:
        out["shear_stress"] = np.asarray(shear_stress, dtype=np.float32)
    if velocity is not None:
        out["velocity"] = np.asarray(velocity, dtype=np.float32)
    if turbulent_viscosity is not None:
        out["turbulent_viscosity"] = np.asarray(turbulent_viscosity, dtype=np.float32)
    for k, v in extra.items():
        if v is not None:
            out[k] = np.asarray(v, dtype=np.float32)
    return out


@dataclass
class FieldDistribution:
    """Per-point/-cell predictive distribution for one canonical field.

    All arrays are in physical (de-normalized) units — the same space as the
    deterministic predictions and ground truth. UQ wrappers denormalize ``mean`` and
    the std/variance channels before returning, so metrics never touch normalization
    stats. ``mean`` / ``std`` may be NumPy arrays or (for the on-device metric path)
    tensors such as ``torch.Tensor``; UQ metrics convert to NumPy at their
    boundary.

    The ``mean`` / ``std`` pair is the Gaussian summary every UQ method can provide
    and that the Gaussian metrics (NLPD, zRMS, coverage) consume. ``samples`` and
    ``quantiles`` / ``quantile_levels`` are optional non-Gaussian escape hatches for
    methods whose predictive law is not Gaussian (quantile regression, conformal,
    evidential, generative); distribution-agnostic metrics (CRPS, quantile coverage) use
    them when present.
    """

    mean: ArrayLike  # (N,) or (N, C)
    std: ArrayLike | None = None  # total predictive std (epistemic + aleatoric)
    epistemic_std: ArrayLike | None = None
    aleatoric_std: ArrayLike | None = None
    samples: ArrayLike | None = None  # optional (S, N[, C]) for CRPS / non-Gaussian
    quantiles: ArrayLike | None = (
        None  # optional (Q, N[, C]) values at ``quantile_levels``
    )
    quantile_levels: ArrayLike | None = (
        None  # optional (Q,) in (0, 1), for interval methods
    )


class FieldDistributionValidationError(ValueError):
    """A malformed :class:`FieldDistribution` payload (shape / non-finite / negative-std defect).

    Subclasses :exc:`ValueError` so existing ``except ValueError`` handlers keep working, while
    giving the benchmark engine a precise seam to recover **per case** without swallowing
    unrelated ``ValueError``\\ s (e.g. configuration errors)."""


def _distribution_shape(x: ArrayLike) -> tuple[int, ...] | None:
    """Best-effort array shape for a NumPy array or framework tensor (``None`` if unavailable)."""
    shape = getattr(x, "shape", None)
    if shape is None:
        return None
    return tuple(int(d) for d in shape)


def _check_finite_nonneg(arr: ArrayLike, name: str, *, nonneg: bool) -> None:
    """Reject non-finite (and, when ``nonneg``, negative) entries for NumPy payloads.

    Value checks run only for NumPy arrays: the ``build_predictive_distribution`` contract keeps
    framework tensors on-device, so we deliberately do not force a host sync to inspect them here
    (metrics validate them again at their NumPy conversion boundary).
    """
    if not isinstance(arr, np.ndarray):
        return
    if not np.all(np.isfinite(arr)):
        raise FieldDistributionValidationError(
            f"FieldDistribution.{name} contains non-finite values (NaN/Inf)."
        )
    if nonneg and np.any(arr < 0.0):
        raise FieldDistributionValidationError(
            f"FieldDistribution.{name} contains negative values; std must be >= 0."
        )


def validate_field_distribution(
    dist: FieldDistribution, *, expected_num_points: int | None = None
) -> None:
    """Fail loudly on a malformed :class:`FieldDistribution` instead of scoring misleading values.

    Checks the per-field contract the UQ metrics assume: ``mean`` is a 1-D ``(N,)`` or 2-D
    ``(N, C)`` array; every provided std channel matches ``mean``'s shape exactly; and (for NumPy
    payloads) ``mean`` is finite while the std channels are finite and non-negative. When
    ``expected_num_points`` is given (e.g. the mesh point/cell count the caller knows), the leading
    dimension must equal it; left ``None`` the check is skipped so the boundary stays
    workflow-agnostic. Defects raise :exc:`FieldDistributionValidationError` (a
    :exc:`ValueError` subclass).

    **Abort-policy split** — who handles a defect where:

    - **Builder (here / :func:`build_predictive_distribution`)**: fail loud *per case* at
      construction, so a bad payload never reaches the metrics as a misleading number.
    - **Benchmark engine** (``benchmarks.engine._run_single``): catches
      :exc:`FieldDistributionValidationError` around the per-case inference stage and recovers —
      that case's configured metrics are recorded as NaN with a ``distribution_validation_error``
      traceback on the per-case row (audited in ``benchmark_artifacts.json``) and the sweep
      continues; ``run.fail_on_any_metric_nan`` remains the opt-in hard failure.
    - **UQ metric guards** (``metrics/builtin/uq.py``): skip-and-warn backstop for payloads this
      validator cannot value-check — framework tensors (kept on device here; converted and
      re-checked at the metrics' NumPy boundary) and legacy :class:`FieldDistribution` objects
      built without this validator.
    """
    shape = _distribution_shape(dist.mean)
    if shape is None or len(shape) not in (1, 2):
        raise FieldDistributionValidationError(
            f"FieldDistribution.mean must be a 1-D (N,) or 2-D (N, C) array; got shape {shape!r}."
        )
    if expected_num_points is not None and shape[0] != expected_num_points:
        raise FieldDistributionValidationError(
            f"FieldDistribution.mean has {shape[0]} rows but the mesh has "
            f"{expected_num_points}; mean/std must be one value per mesh point or cell."
        )
    _check_finite_nonneg(dist.mean, "mean", nonneg=False)
    for name in ("std", "epistemic_std", "aleatoric_std"):
        arr = getattr(dist, name)
        if arr is None:
            continue
        ashape = _distribution_shape(arr)
        if ashape != shape:
            raise FieldDistributionValidationError(
                f"FieldDistribution.{name} shape {ashape!r} must match mean shape {shape!r}."
            )
        _check_finite_nonneg(arr, name, nonneg=True)


def build_predictive_distribution(
    *,
    mean: ArrayLike,
    std: ArrayLike | None = None,
    epistemic_std: ArrayLike | None = None,
    aleatoric_std: ArrayLike | None = None,
    samples: ArrayLike | None = None,
    quantiles: ArrayLike | None = None,
    quantile_levels: ArrayLike | None = None,
    validate: bool = True,
    expected_num_points: int | None = None,
) -> FieldDistribution:
    """Build a :class:`FieldDistribution`, mirroring :func:`build_predictions_dict`.

    Use this in UQ wrappers' :meth:`~physicsnemo.cfd.evaluation.models.model_registry.CFDModel.decode_distribution`
    (prefer keyword arguments) so the predictive-distribution payload stays consistent.
    NumPy inputs are coerced to ``float32``; framework tensors (e.g. ``torch.Tensor``) are
    passed through untouched so the on-device metric path can keep them on device.

    ``validate`` (default on) runs :func:`validate_field_distribution` on the built payload so a
    shape/finiteness/negative-std defect fails at construction rather than surfacing as a
    misleading metric; pass ``expected_num_points`` (the mesh point/cell count) to also assert the
    leading dimension. Set ``validate=False`` to skip the check.
    """

    def _coerce(x: ArrayLike | None) -> ArrayLike | None:
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x.astype(np.float32, copy=False)
        return x  # tensor or other array-like: leave as-is

    dist = FieldDistribution(
        mean=_coerce(mean),
        std=_coerce(std),
        epistemic_std=_coerce(epistemic_std),
        aleatoric_std=_coerce(aleatoric_std),
        samples=_coerce(samples),
        quantiles=_coerce(quantiles),
        quantile_levels=_coerce(quantile_levels),
    )
    if validate:
        validate_field_distribution(dist, expected_num_points=expected_num_points)
    return dist


def as_distribution(predictions: dict[str, Any], key: str) -> FieldDistribution | None:
    """Return the predictive distribution for ``key`` from a predictions dict.

    - A :class:`FieldDistribution` value is returned as-is.
    - A plain array value is wrapped as a **degenerate** distribution (``std=None``) so
      UQ metrics can treat deterministic and probabilistic wrappers uniformly.
    - A missing / ``None`` value returns ``None``.
    """
    value = predictions.get(key)
    if value is None:
        return None
    if isinstance(value, FieldDistribution):
        return value
    return FieldDistribution(mean=value)


def distribution_mean(value: Any) -> Any:
    """Unwrap the point estimate from a predictions value.

    Returns ``value.mean`` for a :class:`FieldDistribution`, else ``value`` unchanged.
    Deterministic metrics use this so a probabilistic wrapper's :class:`FieldDistribution`
    scores through the existing (mean-based) L2 / force / physics metrics.
    """
    if isinstance(value, FieldDistribution):
        return value.mean
    return value
