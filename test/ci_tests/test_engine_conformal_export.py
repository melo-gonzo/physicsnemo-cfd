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

"""Unit tests for the benchmark engine's conformal-export machinery and per-case recovery.

Covers the private helpers directly (bool coercion, export tri-state, npz path/writer, LRU case
cache, metric-signature introspection) plus the engine's per-case recovery from a
``FieldDistributionValidationError`` raised on the inference stage.
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from physicsnemo.cfd.evaluation.benchmarks.engine import (
    BenchmarkPolicyError,
    _BoundedCaseCache,
    _accepts_all_extended_kwargs,
    _coerce_optional_bool,
    _conformal_export_enabled,
    _conformal_inputs_path,
    _expected_num_points_for_case,
    _is_unexpected_kwarg_typeerror,
    _write_conformal_inputs,
    run_benchmark,
)
from physicsnemo.cfd.evaluation.config import (
    Config,
    DatasetConfig,
    ModelConfig,
    RunConfig,
    UQConfig,
)
from physicsnemo.cfd.evaluation.datasets.adapter_registry import (
    DatasetAdapter,
    register_adapter,
)
from physicsnemo.cfd.evaluation.datasets.schema import (
    CanonicalCase,
    FieldDistribution,
    build_predictive_distribution,
    distribution_mean,
)
from physicsnemo.cfd.evaluation.metrics import register_metric
from physicsnemo.cfd.evaluation.models.model_registry import (
    CFDModel,
    register_model,
)

# --------------------------------------------------------------------------------------------
# _coerce_optional_bool: CLI strings must not be truthy by accident (bool("false") is True)
# --------------------------------------------------------------------------------------------


def test_coerce_optional_bool_real_bools_and_none() -> None:
    """Real bools pass through; ``None`` coerces falsy (callers gate None before calling)."""
    assert _coerce_optional_bool(True) is True
    assert _coerce_optional_bool(False) is False
    assert _coerce_optional_bool(None) is False


def test_coerce_optional_bool_falsey_strings() -> None:
    """Un-coerced CLI strings 'false'/'0'/'off'/'no'/'' read as False despite bool('false') being True."""
    assert bool("false") is True  # the trap the helper exists to avoid
    for s in ("false", "False", " FALSE ", "0", "off", "OFF", "no", ""):
        assert _coerce_optional_bool(s) is False, s


def test_coerce_optional_bool_truthy_strings() -> None:
    """String forms 'true' / '1' (and any non-falsey token) read as True."""
    for s in ("true", "True", "1", "yes", "on"):
        assert _coerce_optional_bool(s) is True, s


# --------------------------------------------------------------------------------------------
# _conformal_export_enabled tri-state
# --------------------------------------------------------------------------------------------

_CONFORMAL_METRICS = [("conformal_diagnostic", {}), ("l2_pressure", {})]
_PLAIN_METRICS = [("l2_pressure", {})]


def test_conformal_export_enabled_explicit_override_wins() -> None:
    """An explicit True/False ``run.conformal_export`` overrides the metric-derived default."""
    assert _conformal_export_enabled(RunConfig(conformal_export=True), _PLAIN_METRICS)
    assert not _conformal_export_enabled(
        RunConfig(conformal_export=False), _CONFORMAL_METRICS
    )


def test_conformal_export_enabled_none_defaults_from_metrics() -> None:
    """Unset (None) flag defaults on iff a conformal metric is configured."""
    assert _conformal_export_enabled(
        RunConfig(conformal_export=None), _CONFORMAL_METRICS
    )
    assert _conformal_export_enabled(
        RunConfig(conformal_export=None), [("conformal_crc", {})]
    )
    assert not _conformal_export_enabled(
        RunConfig(conformal_export=None), _PLAIN_METRICS
    )


def test_conformal_export_enabled_uncoerced_cli_string_disables() -> None:
    """A raw CLI string 'false' reaching the engine still disables the export."""
    run_cfg = SimpleNamespace(conformal_export="false")
    assert not _conformal_export_enabled(run_cfg, _CONFORMAL_METRICS)
    assert _conformal_export_enabled(
        SimpleNamespace(conformal_export="1"), _PLAIN_METRICS
    )


# --------------------------------------------------------------------------------------------
# _conformal_inputs_path: dataset-label stem prefix + sanitization
# --------------------------------------------------------------------------------------------


def test_conformal_inputs_path_label_prefix_and_fallback(tmp_path: Path) -> None:
    """The dataset label prefixes the stem; without a label the stem is the case id alone."""
    p = _conformal_inputs_path(str(tmp_path), "modelA", "run_1", "estateback")
    assert p == tmp_path / "conformal_inputs" / "modelA" / "estateback_run_1.npz"
    p_nolabel = _conformal_inputs_path(str(tmp_path), "modelA", "run_1", None)
    assert p_nolabel == tmp_path / "conformal_inputs" / "modelA" / "run_1.npz"


def test_conformal_inputs_path_sanitizes_tokens(tmp_path: Path) -> None:
    """Path separators / whitespace in model, label, and case tokens are replaced, not nested."""
    p = _conformal_inputs_path(
        str(tmp_path), "geo/transolver v2", "run 1?", "fast back"
    )
    assert p.parent.name == "geo_transolver_v2"
    assert p.name == "fast_back_run_1_.npz"
    assert p.parent.parent == tmp_path / "conformal_inputs"


def test_conformal_inputs_path_distinct_for_shared_case_ids(tmp_path: Path) -> None:
    """Two datasets reusing case id 'run_1' get two distinct npz paths (no overwrite)."""
    p_est = _conformal_inputs_path(str(tmp_path), "m", "run_1", "estateback")
    p_fast = _conformal_inputs_path(str(tmp_path), "m", "run_1", "fastback")
    assert p_est != p_fast
    assert {p_est.name, p_fast.name} == {"estateback_run_1.npz", "fastback_run_1.npz"}


# --------------------------------------------------------------------------------------------
# _write_conformal_inputs: field selection, std companions, empty-gt contract, dataset key
# --------------------------------------------------------------------------------------------


def _write_case(
    tmp_path: Path, predictions: dict, gt: dict, *, label="est", points=None
):
    """Invoke the writer with a minimal fake case and return the loaded npz payload dict."""
    ref_geo = SimpleNamespace(points=points) if points is not None else None
    case = SimpleNamespace(reference_geometry=ref_geo)
    err = _write_conformal_inputs(
        output_dir=str(tmp_path),
        model_name="m",
        case_id="run_1",
        predictions=predictions,
        gt=gt,
        case=case,
        dataset_name="ds",
        dataset_label=label,
    )
    assert err is None
    effective = (
        label if label else "ds"
    )  # writer falls back to dataset_name when unlabeled
    with np.load(_conformal_inputs_path(str(tmp_path), "m", "run_1", effective)) as z:
        return {k: z[k] for k in z.files}


def test_write_conformal_inputs_plain_and_distribution_fields(tmp_path: Path) -> None:
    """pred_/true_ are written per overlapping field; std_<field> only for distributions with std."""
    preds = {
        "pressure": np.arange(4, dtype=np.float32),  # plain array -> no std companion
        "shear_stress": FieldDistribution(
            mean=np.ones((4, 3), np.float32), std=np.full((4, 3), 0.1, np.float32)
        ),
        "velocity": FieldDistribution(
            mean=np.ones(4, np.float32)
        ),  # std=None -> no std key
        "extra": np.ones(4, np.float32),  # not in gt -> skipped entirely
    }
    gt = {
        "pressure": np.zeros(4, np.float32),
        "shear_stress": np.zeros((4, 3), np.float32),
        "velocity": np.zeros(4, np.float32),
    }
    data = _write_case(tmp_path, preds, gt)
    assert set(data) == {
        "pred_pressure",
        "true_pressure",
        "pred_shear_stress",
        "true_shear_stress",
        "std_shear_stress",
        "pred_velocity",
        "true_velocity",
        "dataset",
    }
    np.testing.assert_allclose(data["pred_pressure"], np.arange(4))
    np.testing.assert_allclose(data["std_shear_stress"], 0.1)
    assert data["dataset"].item() == "est"


def test_write_conformal_inputs_attaches_points_from_reference_geometry(
    tmp_path: Path,
) -> None:
    """When the case exposes reference geometry, its points array is stored alongside the fields."""
    pts = np.random.default_rng(0).normal(size=(4, 3)).astype(np.float32)
    preds = {"pressure": np.ones(4, np.float32)}
    gt = {"pressure": np.zeros(4, np.float32)}
    data = _write_case(tmp_path, preds, gt, points=pts)
    np.testing.assert_allclose(data["points"], pts)


def test_write_conformal_inputs_empty_gt_writes_marker_npz(tmp_path: Path) -> None:
    """Empty ground truth still writes the per-case npz (cache-hit contract) with only the dataset key."""
    data = _write_case(tmp_path, {"pressure": np.ones(2, np.float32)}, {})
    assert set(data) == {"dataset"}  # no pred_/true_ fields: the reader skips this file
    assert data["dataset"].item() == "est"


def test_write_conformal_inputs_dataset_key_in_every_npz(tmp_path: Path) -> None:
    """Every npz carries a ``dataset`` key: the label, falling back to dataset_name when unlabeled."""
    preds = {"pressure": np.ones(3, np.float32)}
    gt = {"pressure": np.zeros(3, np.float32)}
    labeled = _write_case(tmp_path, preds, gt, label="fastback")
    assert labeled["dataset"].item() == "fastback"
    # No label -> falls back to the (required) dataset name, so two datasets sharing a case id
    # can never collide on the same stem or reader group.
    unlabeled = _write_case(tmp_path, preds, gt, label=None)
    assert unlabeled["dataset"].item() == "ds"


# --------------------------------------------------------------------------------------------
# _BoundedCaseCache LRU semantics
# --------------------------------------------------------------------------------------------


def test_bounded_case_cache_evicts_lru_past_maxsize() -> None:
    """Insertion beyond maxsize drops the least-recently-used entry."""
    cache = _BoundedCaseCache(2)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    assert "a" not in cache
    assert set(cache) == {"b", "c"}


def test_bounded_case_cache_getitem_refreshes_recency() -> None:
    """Reading an entry marks it most-recently-used, so it survives the next eviction."""
    cache = _BoundedCaseCache(2)
    cache["a"] = 1
    cache["b"] = 2
    assert cache["a"] == 1  # refresh 'a'
    cache["c"] = 3
    assert "a" in cache and "b" not in cache


def test_bounded_case_cache_reinsert_does_not_evict() -> None:
    """Overwriting an existing key updates value + recency without evicting anything."""
    cache = _BoundedCaseCache(2)
    cache["a"] = 1
    cache["b"] = 2
    cache["a"] = 10  # re-insert, no growth
    assert len(cache) == 2 and cache["a"] == 10
    cache["c"] = 3  # now 'b' (LRU after the refresh) is evicted
    assert set(cache) == {"a", "c"}


def test_bounded_case_cache_zero_size_disables_via_run_benchmark_wiring() -> None:
    """``matrix_case_cache_size=0`` disables caching (None) as wired; the class itself clamps to 1."""
    for cache_size, expect_none in ((0, True), (1, False)):
        cache = (
            _BoundedCaseCache(cache_size) if cache_size > 0 else None
        )  # engine wiring
        assert (cache is None) is expect_none
    clamped = _BoundedCaseCache(0)  # direct construction clamps maxsize to >= 1
    clamped["a"] = 1
    clamped["b"] = 2
    assert len(clamped) == 1 and "b" in clamped


# --------------------------------------------------------------------------------------------
# Metric-signature introspection (_accepts_all_extended_kwargs / _is_unexpected_kwarg_typeerror)
# --------------------------------------------------------------------------------------------


def test_accepts_all_extended_kwargs_plain_fn_false() -> None:
    """A legacy ``(gt, predictions)`` signature does not accept the extended kwargs."""

    def legacy(gt, predictions):
        """Legacy two-argument metric stub."""
        return 0.0

    assert _accepts_all_extended_kwargs(legacy) is False


def test_accepts_all_extended_kwargs_var_keyword_true() -> None:
    """Declaring ``**kwargs`` (or naming every extended key) accepts the extended call."""

    def modern(gt, predictions, **kwargs):
        """Metric stub declaring ``**kwargs``."""
        return 0.0

    def named(gt, predictions, *, case, comparison_mesh, metric_dtype, output):
        """Metric stub naming every extended kwarg explicitly."""
        return 0.0

    assert _accepts_all_extended_kwargs(modern) is True
    assert _accepts_all_extended_kwargs(named) is True


def test_accepts_all_extended_kwargs_unintrospectable_returns_none() -> None:
    """A C callable without a signature (e.g. ``min``) hits the None branch (try-then-fallback)."""
    assert _accepts_all_extended_kwargs(min) is None
    # Note: ``len`` and ``functools.partial`` ARE introspectable in CPython 3.11+ — they resolve
    # to True/False from the (partial) signature rather than the None branch.
    assert _accepts_all_extended_kwargs(len) is False

    def modern(gt, predictions, **kwargs):
        """Metric stub declaring ``**kwargs``."""
        return 0.0

    def legacy(gt, predictions, extra=None):
        """Legacy metric stub without the extended kwargs."""
        return 0.0

    assert _accepts_all_extended_kwargs(functools.partial(modern)) is True
    assert _accepts_all_extended_kwargs(functools.partial(legacy, extra=1)) is False


def test_is_unexpected_kwarg_typeerror_matches_only_signature_mismatch() -> None:
    """Only the CPython unexpected-keyword TypeError triggers the legacy-call fallback."""

    def fn(a):
        """Single-positional-argument stub."""
        return a

    with pytest.raises(TypeError) as exc:
        fn(1, bogus=2)
    assert _is_unexpected_kwarg_typeerror(exc.value) is True
    assert (
        _is_unexpected_kwarg_typeerror(TypeError("bad operand type for abs()")) is False
    )
    with pytest.raises(TypeError) as arity_exc:
        fn(1, 2)  # arity error, different message
    assert _is_unexpected_kwarg_typeerror(arity_exc.value) is False


# --------------------------------------------------------------------------------------------
# _expected_num_points_for_case: engine-side dof count for distribution validation
# --------------------------------------------------------------------------------------------


def test_expected_num_points_for_case_uses_output_location() -> None:
    """Point-output wrappers validate against n_points, cell-output wrappers against n_cells."""
    case = SimpleNamespace(reference_geometry=SimpleNamespace(n_points=10, n_cells=6))
    assert (
        _expected_num_points_for_case(case, SimpleNamespace(output_location="point"))
        == 10
    )
    assert (
        _expected_num_points_for_case(case, SimpleNamespace(output_location="cell"))
        == 6
    )


def test_expected_num_points_for_case_none_without_geometry() -> None:
    """No loaded reference geometry -> None (check skipped; no extra mesh read to count dof)."""
    case = SimpleNamespace(reference_geometry=None)
    assert (
        _expected_num_points_for_case(case, SimpleNamespace(output_location="point"))
        is None
    )
    bare_geo = SimpleNamespace()  # geometry without n_points/n_cells attributes
    assert (
        _expected_num_points_for_case(
            SimpleNamespace(reference_geometry=bare_geo),
            SimpleNamespace(output_location="point"),
        )
        is None
    )


# --------------------------------------------------------------------------------------------
# Per-case engine recovery from FieldDistributionValidationError (FIX 4)
# --------------------------------------------------------------------------------------------


class _RecoveryAdapter(DatasetAdapter):
    """Two-case in-memory adapter; mesh path is intentionally bogus (no mesh I/O required)."""

    def __init__(self, root: str = "", **kwargs: Any) -> None:
        self._root = root

    def list_cases(self) -> list[str]:
        """Return the fixed bad-then-good case pair."""
        return ["case_bad", "case_good"]

    def load_case(self, case_id: str) -> CanonicalCase:
        """Build an in-memory canonical case with a 4-point pressure ground truth."""
        return CanonicalCase(
            case_id=case_id,
            mesh_path="/nonexistent/does_not_exist.vtp",
            mesh_type="point",
            ground_truth={"pressure": np.arange(4, dtype=np.float32)},
            inference_domain="surface",
        )


class _NanDistributionWrapper(CFDModel):
    """Closed-form UQ wrapper whose decode builds a NaN-mean distribution for ``case_bad``."""

    OUTPUT_LOCATION = "point"
    INFERENCE_DOMAIN = "surface"
    REQUIRES_REMOTE_ASSETS = False
    SUPPORTS_UQ = True
    UQ_METHOD = "closed_form"

    @property
    def output_location(self) -> str:
        """Predictions live on mesh points."""
        return "point"

    def load(self, checkpoint_path, stats_path, device, **kwargs):
        """No weights to load for this stub."""
        return self

    def prepare_inputs(self, case):
        """The case itself is the model input."""
        return case

    def predict(self, model_input):
        """Forward pass is the identity for this stub."""
        return model_input

    def decode_outputs(self, raw, case, model_input=None):
        """Deterministic decode (unused on the closed-form path)."""
        return {"pressure": np.asarray(case.ground_truth["pressure"], np.float32)}

    def decode_distribution(self, raw, case, model_input=None):
        """Return a perfect distribution for ``case_good``; fail validation for ``case_bad``."""
        if case.case_id == "case_bad":
            mean = np.full(4, np.nan, dtype=np.float32)  # validation raises here
        else:
            mean = np.asarray(case.ground_truth["pressure"], np.float32)
        return {
            "pressure": build_predictive_distribution(
                mean=mean, std=np.full(4, 0.1, np.float32)
            )
        }


def _probe_mae(gt: dict, predictions: dict, **kwargs: Any) -> float:
    """Mean absolute pressure error against ground truth (mesh-free probe metric)."""
    pred = np.asarray(distribution_mean(predictions["pressure"]))
    return float(np.mean(np.abs(pred - np.asarray(gt["pressure"]))))


register_adapter("_engine_recovery_ds", _RecoveryAdapter)
register_model("_engine_recovery_model", _NanDistributionWrapper)
register_metric("_engine_recovery_mae", _probe_mae, domain="surface")


def _recovery_config(tmp_path: Path, **run_overrides: Any) -> Config:
    """Benchmark config wiring the recovery stubs on CPU with UQ enabled."""
    return Config(
        run=RunConfig(
            device="cpu",
            output_dir=str(tmp_path),
            save_inference_mesh=False,
            uq=UQConfig(enabled=True),
            **run_overrides,
        ),
        model=ModelConfig(name="_engine_recovery_model", checkpoint="", stats_path=""),
        dataset=DatasetConfig(name="_engine_recovery_ds", root=str(tmp_path)),
        metrics=["_engine_recovery_mae"],
    )


def test_engine_recovers_per_case_from_distribution_validation_error(
    tmp_path: Path,
) -> None:
    """One NaN-mean case marks only that case failed; the run completes and the good case scores."""
    results = run_benchmark(_recovery_config(tmp_path))
    assert len(results) == 1
    res = results[0]
    rows = {r["case_id"]: r for r in res["per_case"]}
    assert set(rows) == {"case_bad", "case_good"}
    bad, good = rows["case_bad"], rows["case_good"]
    # Bad case: configured metric NaN + full traceback recorded on the row.
    assert math.isnan(bad["metrics"]["_engine_recovery_mae"])
    assert "non-finite" in bad["distribution_validation_error"]
    # Good case: scored normally, no error key.
    assert good["metrics"]["_engine_recovery_mae"] == 0.0
    assert "distribution_validation_error" not in good
    # Aggregate filters the failed case's NaN (mirrors recoverable-metric behavior).
    assert res["metrics"]["_engine_recovery_mae"] == 0.0
    # Audit entry lands in benchmark_artifacts.json, mirroring conformal_export_failures.
    artifacts = json.loads((tmp_path / "benchmark_artifacts.json").read_text())
    failures = artifacts["distribution_validation_failures"]
    assert [e["case_id"] for e in failures] == ["case_bad"]
    assert "non-finite" in failures[0]["traceback"]


def test_engine_recovery_respects_fail_on_any_metric_nan(tmp_path: Path) -> None:
    """``run.fail_on_any_metric_nan`` opts into hard failure when recovery leaves a NaN aggregate."""
    cfg = _recovery_config(tmp_path, fail_on_any_metric_nan=True)
    cfg.dataset.case_ids = ["case_bad"]  # every case fails -> aggregate metric is NaN
    with pytest.raises(BenchmarkPolicyError, match="fail_on_any_metric_nan"):
        run_benchmark(cfg)
