import pandas as pd
import pytest

from src import selection


def rows(name, f1s, section=5):
    return [{"section": section, "cfg_name": name, "seed": 11 * (i + 1), "val_macro_f1": f1, "val_f1_rainy": f1,
             "epochs_run": 40, "best_epoch": 25, "minutes": 1.0} for i, f1 in enumerate(f1s)]


def results(*groups):
    return pd.DataFrame([row for group in groups for row in group])


def test_summary_uses_sample_std_and_skips_smoke():
    summary = selection.config_summary(results(rows("a", [0.9, 0.92, 0.94]), rows("smoke_x", [0.1])))
    assert list(summary["cfg_name"]) == ["a"]
    assert summary.loc[0, "f1_mean"] == pytest.approx(0.92) and summary.loc[0, "f1_std"] == pytest.approx(0.02)
    assert summary.loc[0, "f1_min"] == 0.9 and summary.loc[0, "seeds"] == 3


def test_worst_seed_floor_and_seed_count_gate_selection():
    summary = selection.config_summary(results(rows("high_but_one_bad_seed", [0.99, 0.99, 0.99, 0.99, 0.80]),
                                               rows("three_seeds", [0.97, 0.97, 0.97]),
                                               rows("steady", [0.90, 0.91, 0.92, 0.90, 0.91])))
    assert selection.select_config(summary)["cfg_name"] == "steady"


def test_tie_within_one_std_goes_to_lower_std():
    summary = selection.config_summary(results(rows("noisy", [0.88, 0.98, 0.90, 0.97, 0.94]),
                                               rows("calm", [0.93, 0.93, 0.94, 0.93, 0.93])))
    assert selection.select_config(summary)["cfg_name"] == "calm"


def test_clear_winner_beats_lower_std():
    summary = selection.config_summary(results(rows("better", [0.95, 0.96, 0.94, 0.95, 0.96]),
                                               rows("calm", [0.90, 0.90, 0.90, 0.90, 0.90])))
    assert selection.select_config(summary)["cfg_name"] == "better"


def test_no_eligible_config_returns_none():
    summary = selection.config_summary(results(rows("weak", [0.5, 0.6, 0.7, 0.8, 0.9])))
    assert selection.select_config(summary) is None
