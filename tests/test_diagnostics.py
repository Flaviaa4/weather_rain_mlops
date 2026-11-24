import subprocess
import sys

import pytest

from src import diagnostics

PASSING_PIPELINE = {"passed": True}


def row(name, train, inference, refreshed, paths_same=True):
    manual = {"accuracy": inference, "loss": 1.0}
    return {"variant": name, "training_mode": {"accuracy": train, "loss": 0.01},
            "inference": {"evaluate": {"accuracy": inference}, "predict": manual,
                          "manual": {"accuracy": inference if paths_same else inference + 0.1, "loss": 1.0},
                          "predict_vs_manual_max_abs_diff": 0.0},
            "inference_after_bn_refresh": {"accuracy": refreshed, "loss": 0.01}}


def rows(check, no_bn=(1.0, 1.0, 1.0), r0=(1.0, 1.0, 1.0), r0_plain=(1.0, 1.0, 1.0), paths_same=True):
    return [row("baseline", 1.0, 1.0, 1.0), row("r0", *r0), row("r0_no_batch_norm", *no_bn),
            row("r0_dropout0_no_augment", *r0_plain), row("r0_bn_momentum_0.9", 1.0, 0.9, 1.0),
            row(diagnostics.CHECK_VARIANT, *check, paths_same=paths_same)]


def test_batch_norm_lag_is_cause_a():
    assert diagnostics.verdict(PASSING_PIPELINE, rows(check=(1.0, 0.406, 1.0)))[0] == "a"


def test_gap_that_survives_refresh_is_a_bug():
    assert diagnostics.verdict(PASSING_PIPELINE, rows(check=(1.0, 0.406, 0.5)))[0] == "c"


def test_broken_pipeline_or_disagreeing_paths_is_a_bug():
    assert diagnostics.verdict({"passed": False}, rows(check=(1.0, 0.406, 1.0)))[0] == "c"
    assert diagnostics.verdict(PASSING_PIPELINE, rows(check=(1.0, 0.406, 1.0), paths_same=False))[0] == "c"


def test_gap_only_with_dropout_or_augmentation_is_cause_b():
    assert diagnostics.verdict(PASSING_PIPELINE, rows(check=(1.0, 1.0, 1.0), r0=(0.9, 0.6, 0.9)))[0] == "b"


def test_unfitted_check_is_not_called_batch_norm():
    assert diagnostics.verdict(PASSING_PIPELINE, rows(check=(0.5, 0.3, 0.5)))[0] == "c"


def test_diagnostics_module_does_not_import_tensorflow():
    code = "import sys, src.diagnostics; print('tensorflow' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"
