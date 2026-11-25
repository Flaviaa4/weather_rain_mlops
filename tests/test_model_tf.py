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


def random_batch(count=16, size=32):
    rng = np.random.default_rng(0)
    return rng.random((count, size, size, 3), dtype=np.float32), metrics.one_hot(np.tile(np.arange(4), count // 4), 4)


def test_baseline_network_matches_section3_parameter_count():
    built = model.build_model(grids.BASELINE)
    assert built.count_params() == 8_408_644
    assert [layer.__class__.__name__ for layer in built.layers].count("Conv2D") == 2


def test_ablation_and_momentum_options_reach_the_layers():
    with_bn = model.build_model(grids.variant(grids.BASELINE, "bn", batch_norm=True))
    assert len(model.batch_norm_layers(with_bn)) == 2
    assert all(layer.momentum == 0.99 for layer in model.batch_norm_layers(with_bn))
    fast = model.build_model(grids.variant(small(grids.R0), "m", bn_momentum=0.9))
    assert all(layer.momentum == 0.9 for layer in model.batch_norm_layers(fast))
    pooled = model.build_model(grids.variant(grids.BASELINE, "gap", pooling="gap"))
    assert "GlobalAveragePooling2D" in {layer.__class__.__name__ for layer in pooled.layers}


def test_warmup_rates_and_best_weights_after_min_epochs(tmp_path):
    cfg = small(grids.R0, max_epochs=3, warmup_epochs=2, min_epochs=2, early_stop_patience=10)
    images, targets = random_batch()
    built = model.compile_model(model.build_model(cfg), cfg)
    built.fit(images, targets, validation_data=(images, targets), epochs=3,
              callbacks=model.get_callbacks(cfg, tmp_path / "history.csv"), verbose=0)
    history = experiments.run_report.read_history(tmp_path / "history.csv")
    assert history["learning_rate"].tolist() == pytest.approx([5e-4, 1e-3, 1e-3], rel=1e-5)


@pytest.mark.parametrize("start, expected", [(0, 0), (2, 3), (9, 0)])
def test_restore_best_weights_ignores_epochs_before_start(start, expected):
    built = model.build_model(small(grids.BASELINE))
    callback = model.RestoreBestWeights(start_epoch=start)
    callback.set_model(built)
    callback.on_train_begin()
    for epoch, loss in enumerate([0.1, 0.2, 0.5, 0.4, 0.6]):
        built.set_weights([np.full_like(weights, epoch) for weights in built.get_weights()])
        callback.on_epoch_end(epoch, {"val_loss": loss})
    callback.on_train_end()
    assert built.get_weights()[0].flat[0] == expected


def test_save_without_optimizer_predicts_the_same_and_is_not_larger(tmp_path):
    cfg = small(grids.R0)
    images, targets = random_batch()
    built = model.compile_model(model.build_model(cfg), cfg)
    built.fit(images, targets, epochs=1, verbose=0)
    built.save(tmp_path / "with_optimizer.keras")
    model.save_without_optimizer(built, tmp_path / "model.keras")
    loaded = model.keras.models.load_model(tmp_path / "model.keras")
    assert np.allclose(loaded.predict(images, verbose=0), built.predict(images, verbose=0), atol=1e-5)
    sizes = [(tmp_path / name).stat().st_size for name in ["model.keras", "with_optimizer.keras"]]
    print(f"model.keras without optimizer {sizes[0]} bytes, with optimizer {sizes[1]} bytes")
    assert sizes[0] <= sizes[1]


def test_final_export_writes_h5_and_onnx_with_parity(tmp_path):
    import shutil
    from src import final_model
    from tests.conftest import write_tiny_dataset
    data_dir = write_tiny_dataset(tmp_path / "data", per_class=6)
    shutil.copytree(data_dir / "valid", data_dir / "test")
    ctx = {"section": 7, "group": "test", "data_dir": data_dir, "dir": tmp_path / "exp", "smoke": True, "backup": False}
    cfg = small(grids.final_config())
    experiments.run_one(cfg, 11, ctx)
    record = final_model.run_final(cfg, ctx)
    paths = final_model.output_paths(ctx["dir"])
    assert record["parity"]["validation"]["passed"] and record["parity"]["test"]["n_images"] == 24
    reloaded = model.keras.models.load_model(paths["h5"], compile=False)
    images = experiments.load_split_cached("valid", 32, data_dir)[0]
    onnx_probs = final_model.onnx_probabilities(paths["onnx"], images)
    assert np.abs(final_model.keras_probabilities(reloaded, images) - onnx_probs).max() < final_model.PARITY_TOLERANCE
    with pytest.raises(RuntimeError, match="already evaluated"):
        final_model.run_final(cfg, ctx)


def test_section5_smoke_config_runs_end_to_end(tiny_dataset, tmp_path):
    ctx = {"section": 5, "group": "test", "data_dir": tiny_dataset, "dir": tmp_path, "smoke": True, "backup": False}
    for cfg in grids.smoke_configs(5):
        row = experiments.run_one(small(cfg), 11, ctx)
        assert experiments.check_run_artifacts(tmp_path / "runs" / row["run_id"])
    history = experiments.run_report.read_history(tmp_path / "runs" / "s5_smoke_r0_options_seed11" / "history.csv")
    assert history["learning_rate"].iloc[0] == pytest.approx(1.5e-4, rel=1e-5)
