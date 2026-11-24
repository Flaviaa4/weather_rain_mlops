import json

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src import experiments, grids, metrics, model


def small(cfg, **changes):
    return {**cfg, "image_size": 32, **changes}


def test_r0_shape_and_layers():
    built = model.build_model(grids.R0)
    assert built.input_shape == (None, 128, 128, 3) and built.output_shape == (None, 4)
    names = [layer.__class__.__name__ for layer in built.layers]
    assert names.count("Conv2D") == 6 and "BatchNormalization" in names and "GlobalAveragePooling2D" in names
    assert 200_000 < built.count_params() < 500_000


def test_baseline_has_no_regularization():
    built = model.build_model(grids.BASELINE)
    names = {layer.__class__.__name__ for layer in built.layers}
    assert names.isdisjoint({"BatchNormalization", "Dropout"}) and "Flatten" in names
    assert built.output_shape == (None, 4)


@pytest.mark.parametrize("optimizer", ["adam", "adamw", "sgd", "rmsprop"])
def test_optimizers_build(optimizer):
    assert model.build_optimizer({**grids.R0, "optimizer": optimizer, "weight_decay": 1e-4}) is not None


def test_class_weights_balance_counts():
    weights = model.compute_class_weights(np.array([0, 0, 0, 1, 2, 3]))
    assert weights[0] == pytest.approx(0.5) and weights[1] == pytest.approx(1.5)


def test_restore_best_weights_and_history_columns(tmp_path):
    cfg = small(grids.R0, max_epochs=4, early_stop_patience=10)
    rng = np.random.default_rng(0)
    images, labels = rng.random((16, 32, 32, 3), dtype=np.float32), np.tile(np.arange(4), 4)
    built = model.compile_model(model.build_model(cfg), cfg)
    callbacks = model.get_callbacks(cfg, tmp_path / "history.csv")
    built.fit(images, metrics.one_hot(labels, 4), validation_data=(images, metrics.one_hot(labels, 4)),
              epochs=4, callbacks=callbacks, verbose=0)
    history = experiments.run_report.read_history(tmp_path / "history.csv")
    assert {"loss", "val_loss", "accuracy", "val_accuracy", "learning_rate", "epoch_seconds"} <= set(history)
    restored = built.evaluate(images, metrics.one_hot(labels, 4), verbose=0, return_dict=True)["loss"]
    assert restored == pytest.approx(history["val_loss"].min(), rel=1e-3)


def test_run_one_end_to_end(tiny_dataset, tmp_path):
    ctx = {"section": 3, "group": "test", "data_dir": tiny_dataset, "dir": tmp_path, "smoke": True, "backup": False}
    row = experiments.run_one(small(grids.R0), 11, ctx)
    run_dir = tmp_path / "runs" / "s3_r0_seed11"
    assert experiments.check_run_artifacts(run_dir)
    assert row["epochs_run"] == experiments.SMOKE_EPOCHS and "val_f1_rainy" in row
    assert json.loads((run_dir / "config.json").read_text())["config"]["max_epochs"] == experiments.SMOKE_EPOCHS
    assert experiments.pending_runs([small(grids.R0)], [11], ctx) == []


def test_refresh_batch_norm_makes_inference_match_training_mode():
    cfg = small(grids.R0, dropout=0.0)
    built = model.build_model(cfg)
    images = np.random.default_rng(1).random((8, 32, 32, 3), dtype=np.float32)
    assert model.refresh_batch_norm_statistics(built, images) == 6
    training_mode = built(images, training=True).numpy()
    assert np.allclose(built(images, training=False).numpy(), training_mode, atol=1e-4)
    assert all(layer.momentum == 0.99 for layer in model.batch_norm_layers(built))


def test_diag_runs_and_pipeline_is_identical(tiny_dataset, tmp_path):
    from src import diagnostics
    ctx = {"section": 3, "data_dir": tiny_dataset, "dir": tmp_path, "smoke": True}
    result = diagnostics.run_batch_diagnostics(ctx, steps=2, per_class=3)
    assert result["pipeline"]["passed"], result["pipeline"]
    assert len(result["variants"]) == 6 and result["cause"] in {"a", "b", "c", "not_reproduced"}
    assert all(diagnostics.paths_agree(row) for row in result["variants"])
    assert (tmp_path / "sanity" / "diag_overfit_one_batch.json").exists()
