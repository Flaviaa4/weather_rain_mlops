import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src import config, experiments, grids, metrics, run_report


def fake_history(epochs=6, best=3):
    val_loss = [1.0 - 0.1 * min(i, best) + 0.05 * max(0, i - best) for i in range(epochs)]
    return pd.DataFrame({"epoch": range(epochs), "loss": np.linspace(1, 0.2, epochs), "accuracy": np.linspace(0.4, 0.95, epochs),
                         "val_loss": val_loss, "val_accuracy": np.linspace(0.4, 0.8, epochs),
                         "learning_rate": [1e-3] * 4 + [5e-4] * (epochs - 4), "epoch_seconds": [3.0] + [1.0] * (epochs - 1)})


def fake_row(run_id, cfg):
    return {"run_id": run_id, "config_hash": experiments.config_hash(cfg), "val_macro_f1": 0.5}


def test_experiments_module_does_not_import_tensorflow():
    code = "import sys, src.experiments, src.run_report, src.metrics, src.grids; print('tensorflow' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_run_id_and_hash_are_stable():
    assert experiments.make_run_id(3, grids.R0, 11) == "s3_r0_seed11"
    assert experiments.config_hash(dict(grids.R0)) == experiments.config_hash(grids.R0)
    assert experiments.config_hash(grids.R0) != experiments.config_hash(grids.BASELINE)


def test_baseline_differs_from_r0_only_in_optimizations():
    changed = {key for key in grids.R0 if grids.R0[key] != grids.BASELINE[key]}
    assert changed == {"name", "architecture", "blocks", "convs_per_block", "batch_norm", "dropout", "l2", "pooling",
                       "augment", "class_weights"}
    assert grids.R0["augment"] == "medium" and grids.R0["image_size"] == 128


def test_variant_rejects_unknown_keys():
    assert grids.variant(grids.R0, "lr3e-4", learning_rate=3e-4)["learning_rate"] == 3e-4
    with pytest.raises(KeyError):
        grids.variant(grids.R0, "typo", learnin_rate=1)


def test_log_result_row_and_already_done(tmp_path):
    assert not experiments.already_done("s3_r0_seed11", grids.R0, tmp_path)
    experiments.log_result_row(fake_row("s3_r0_seed11", grids.R0), tmp_path)
    experiments.log_result_row(fake_row("s3_baseline_seed11", grids.BASELINE), tmp_path)
    assert experiments.already_done("s3_r0_seed11", grids.R0, tmp_path)
    assert len(experiments.read_results(tmp_path)) == 2


def test_already_done_refuses_changed_config(tmp_path):
    experiments.log_result_row(fake_row("s3_r0_seed11", grids.R0), tmp_path)
    with pytest.raises(ValueError):
        experiments.already_done("s3_r0_seed11", {**grids.R0, "dropout": 0.5}, tmp_path)


def test_pending_runs_skips_finished(tmp_path):
    ctx = {"section": 3, "dir": tmp_path, "smoke": False}
    experiments.log_result_row(fake_row("s3_r0_seed11", grids.R0), tmp_path)
    pending = experiments.pending_runs([grids.R0], [11, 22, 33], ctx)
    assert [seed for _, seed in pending] == [22, 33]


def test_smoke_config_only_changes_epochs():
    smoke = experiments.effective_config(grids.R0, {"smoke": True})
    assert smoke["max_epochs"] == experiments.SMOKE_EPOCHS
    assert {k for k in smoke if smoke[k] != grids.R0[k]} == {"max_epochs"}


def test_take_per_class_and_stratified_batch():
    images, labels = np.arange(40).reshape(40, 1), np.repeat(np.arange(4), 10)
    _, kept, paths = experiments.take_per_class((images, labels, [str(i) for i in range(40)]), 3)
    assert np.bincount(kept).tolist() == [3, 3, 3, 3] and paths[:3] == ["0", "1", "2"]
    _, batch_labels = experiments.stratified_batch(images, labels, 8, 11)
    assert np.bincount(batch_labels).tolist() == [8, 8, 8, 8]


def test_history_fields_use_best_epoch():
    fields = experiments.history_fields(fake_history(), grids.R0)
    assert fields["best_epoch"] == 4 and fields["epochs_run"] == 6 and fields["stopped_early"]
    assert fields["epoch_seconds_median"] == 1.0 and fields["final_learning_rate"] == 5e-4


def test_read_history_drops_repeated_epochs(tmp_path):
    history = fake_history()
    pd.concat([history, history.iloc[[5]]]).to_csv(tmp_path / "history.csv", index=False)
    assert len(run_report.read_history(tmp_path / "history.csv")) == 6


def test_estimates_and_next_group():
    assert experiments.estimate_total_time(4, 2.5) == 10
    spec = {"configs": [grids.R0, grids.BASELINE], "seeds": [11, 22]}
    minutes = experiments.group_minutes(spec, {"cnn": 2.0, "baseline": 1.0}, 60)
    assert minutes == pytest.approx(2 * (120 + 20) / 60 + 2 * (60 + 20) / 60)
    assert experiments.next_group_message(3, "timing").endswith("--group diag")
    assert experiments.next_group_message(3, "diag").endswith("--group A")
    assert experiments.group_order(3) == ["smoke", "timing", "diag", "A", "B", "C"]
    assert experiments.next_group_message(3, "C") == "Section 3 grid complete."


def test_run_report_writes_files(tmp_path):
    labels = np.array([0, 1, 2, 3, 1])
    probabilities = metrics.one_hot([0, 1, 2, 2, 1], 4) * 0.9 + 0.025
    result = metrics.classification_metrics(probabilities, labels)
    run_report.plot_curves(fake_history(), tmp_path / "curves.png", "test")
    run_report.plot_confusion_matrix(result["confusion_matrix"], tmp_path / "cm.png", "test")
    frame = run_report.save_predictions([f"valid/x{i}.jpg" for i in range(5)], labels, probabilities, tmp_path / "p.csv")
    assert (tmp_path / "curves.png").stat().st_size > 0 and (tmp_path / "cm.png").stat().st_size > 0
    assert frame.loc[3, "true"] == "sunrise" and frame.loc[3, "predicted"] == "shiny"
    assert frame["confidence"].iloc[0] == pytest.approx(0.925)


def test_reference_baselines_file(tiny_dataset, tmp_path):
    path = experiments.run_reference_baselines(tiny_dataset, tmp_path)
    result = json.loads(path.read_text())
    assert {"majority", "random_train_prior", "random_uniform"} <= set(result)


def changed_keys(cfg, base):
    return {key for key in set(cfg) | set(base) if key != "name" and cfg.get(key) != base.get(key)}


def test_section5_arms_change_one_factor_from_their_reference():
    groups = grids.section_groups(5)
    assert list(groups) == ["A", "B", "C", "D"] and experiments.group_order(5) == ["smoke", "timing", "A", "B", "C", "D"]
    r0_arms = groups["A"]["configs"] + groups["B"]["configs"] + groups["D"]["configs"]
    assert [changed_keys(cfg, grids.R0) for cfg in r0_arms] == [
        {"bn_momentum"}, {"bn_momentum"}, {"early_stop_patience", "min_epochs"}, {"learning_rate"},
        {"warmup_epochs"}, {"augment"}]
    ablation = groups["C"]["configs"]
    assert [changed_keys(cfg, grids.BASELINE) for cfg in ablation] == [
        {"batch_norm"}, {"dropout"}, {"l2"}, {"class_weights"}, {"augment"}, {"pooling"}]
    assert all(cfg[key] == grids.R0[key] for cfg in ablation for key in changed_keys(cfg, grids.BASELINE))
    names = [cfg["name"] for spec in groups.values() for cfg in spec["configs"]]
    assert len(names) == len(set(names)) and groups["C"]["seeds"] == [11, 22, 33]


def test_optional_keys_keep_section3_hashes():
    assert set(config.OPTIONAL_CONFIG_DEFAULTS).isdisjoint(grids.R0)
    assert config.config_setting(grids.R0, "bn_momentum") == 0.99 and config.config_setting(grids.R0, "min_epochs") == 0
    assert grids.variant(grids.R0, "m", bn_momentum=0.9)["bn_momentum"] == 0.9
    assert all(cfg["name"].startswith("smoke") for cfg in grids.smoke_configs(5))
    assert [cfg["name"] for cfg in grids.smoke_configs(3)] == ["baseline", "r0", "r0_shuffled_labels"]


def test_best_epoch_respects_min_epochs():
    history = fake_history(epochs=8, best=1)
    assert run_report.best_epoch_index(history) == 1 and run_report.best_epoch_index(history, 4) == 4
    assert run_report.best_epoch_index(history, 20) == 1
    assert experiments.history_fields(history, {**grids.R0, "min_epochs": 4})["best_epoch"] == 5


def result_row(section, cfg, seed, f1, loss=0.5):
    return {"run_id": experiments.make_run_id(section, cfg, seed), "section": section, "seed": seed,
            "config_hash": experiments.config_hash(cfg), "val_macro_f1": f1, "val_loss": loss}


def test_prune_models_keeps_only_best_seed_of_the_config(tmp_path):
    other = grids.variant(grids.R0, "other", learning_rate=3e-4)
    planned = [(5, grids.R0, 11, 0.80), (5, grids.R0, 22, 0.90), (5, grids.R0, 33, 0.85), (5, other, 11, 0.5), (3, grids.R0, 11, 0.7)]
    for section, cfg, seed, f1 in planned:
        experiments.log_result_row(result_row(section, cfg, seed, f1), tmp_path)
        folder = tmp_path / "runs" / experiments.make_run_id(section, cfg, seed)
        folder.mkdir(parents=True)
        (folder / "model.keras").write_bytes(b"x")
    removed = experiments.prune_models(grids.R0, {"dir": tmp_path, "section": 5})
    assert sorted(removed) == ["s5_r0_seed11", "s5_r0_seed33"]
    kept = sorted(path.parent.name for path in tmp_path.glob("runs/*/model.keras"))
    assert kept == ["s3_r0_seed11", "s5_other_seed11", "s5_r0_seed22"]


def test_budget_stops_before_starting_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "run_one", lambda cfg, seed, ctx: pytest.fail("no run may start"))
    ctx = {"section": 5, "dir": tmp_path, "smoke": False, "deadline": 0.0}
    assert experiments.run_grid([grids.R0], [11, 22], ctx) == 0
    monkeypatch.setenv("RUN_BUDGET_MINUTES", "0")
    assert experiments.section_deadline(5) is None
    monkeypatch.delenv("RUN_BUDGET_MINUTES")
    assert experiments.section_deadline(5) is not None and experiments.section_deadline(3) is None
