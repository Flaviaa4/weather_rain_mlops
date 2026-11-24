import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src import config

PLACEHOLDER_SEED = 0
PLACEHOLDER_WARNING = "PLACEHOLDER: random weights, not trained. Replace with the real model from Section 7."
CONV_FILTERS = [8, 16]


def placeholder_metadata():
    return {"placeholder": "true", "warning": PLACEHOLDER_WARNING, "model_name": config.MODEL_NAME,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def he_weights(rng, shape, fan_in):
    return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)


def conv_initializers(rng, prefix, in_channels, out_channels):
    from onnx import numpy_helper
    weights = he_weights(rng, (out_channels, in_channels, 3, 3), in_channels * 9)
    bias = np.zeros(out_channels, dtype=np.float32)
    return [numpy_helper.from_array(weights, f"{prefix}_w"), numpy_helper.from_array(bias, f"{prefix}_b")]


def conv_block_nodes(prefix, source):
    from onnx import helper
    return [helper.make_node("Conv", [source, f"{prefix}_w", f"{prefix}_b"], [f"{prefix}_conv"], pads=[1, 1, 1, 1]),
            helper.make_node("Relu", [f"{prefix}_conv"], [f"{prefix}_relu"]),
            helper.make_node("MaxPool", [f"{prefix}_relu"], [f"{prefix}_out"], kernel_shape=[2, 2], strides=[2, 2])]


def head_nodes(source):
    from onnx import helper
    return [helper.make_node("GlobalAveragePool", [source], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["features"]),
            helper.make_node("Gemm", ["features", "dense_w", "dense_b"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1)]


def head_initializers(rng, in_features, n_classes):
    from onnx import numpy_helper
    weights = he_weights(rng, (in_features, n_classes), in_features)
    return [numpy_helper.from_array(weights, "dense_w"), numpy_helper.from_array(np.zeros(n_classes, np.float32), "dense_b")]


def build_placeholder_graph(size=config.IMAGE_SIZE, n_classes=len(config.CLASS_NAMES), seed=PLACEHOLDER_SEED):
    from onnx import TensorProto, helper
    rng = np.random.default_rng(seed)
    nodes = [helper.make_node("Transpose", ["image"], ["nchw"], perm=[0, 3, 1, 2])]
    initializers, source, channels = [], "nchw", config.CHANNELS
    for index, filters in enumerate(CONV_FILTERS):
        nodes += conv_block_nodes(f"block{index}", source)
        initializers += conv_initializers(rng, f"block{index}", channels, filters)
        source, channels = f"block{index}_out", filters
    nodes += head_nodes(source)
    initializers += head_initializers(rng, channels, n_classes)
    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", size, size, config.CHANNELS])
    output = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["batch", n_classes])
    return helper.make_graph(nodes, "placeholder_random_cnn", [image], [output], initializers)


def build_placeholder_onnx(size=config.IMAGE_SIZE, n_classes=len(config.CLASS_NAMES), seed=PLACEHOLDER_SEED):
    import onnx
    from onnx import helper
    model = helper.make_model(build_placeholder_graph(size, n_classes, seed), opset_imports=[helper.make_opsetid("", 17)],
                              producer_name="weather_rain_mlops.placeholder")
    model.ir_version = 8
    helper.set_model_props(model, placeholder_metadata())
    onnx.checker.check_model(model)
    return model


def save_placeholder_onnx(path=config.ONNX_PATH):
    import onnx
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_placeholder_onnx(), str(path))
    return path


def build_placeholder_keras(h5_path=config.H5_PATH, onnx_path=config.ONNX_PATH):
    from src import grids, model, onnx_export
    keras_model = model.build_model(grids.R0)
    Path(h5_path).parent.mkdir(parents=True, exist_ok=True)
    keras_model.save(str(h5_path))
    onnx_export.export_keras_to_onnx(keras_model, onnx_path, placeholder_metadata())
    return Path(h5_path), Path(onnx_path)


def placeholder_metrics(onnx_path=config.ONNX_PATH, data_dir=config.DATA_DIR):
    from src import prediction
    from src.preprocessing import load_split
    images, labels, _ = load_split("valid", data_dir=data_dir)
    result = prediction.evaluate_onnx_model(images, labels, onnx_path)
    return {"model_name": config.MODEL_NAME, "placeholder": True, "warning": PLACEHOLDER_WARNING,
            "split": "valid", "n_images": int(len(labels)), "validation": result}


def write_placeholder_metrics(onnx_path=config.ONNX_PATH, metrics_path=config.METRICS_PATH, data_dir=config.DATA_DIR):
    from src.run_report import save_json
    return save_json(placeholder_metrics(onnx_path, data_dir), metrics_path)


def ensure_placeholder_files(with_keras):
    if not config.IS_PLACEHOLDER:
        raise RuntimeError(f"MODEL_NAME is {config.MODEL_NAME}, refusing to create a placeholder over it")
    complete = config.ONNX_PATH.exists() and (config.H5_PATH.exists() or not with_keras)
    if not complete:
        build_placeholder_keras(config.H5_PATH, config.ONNX_PATH) if with_keras else save_placeholder_onnx(config.ONNX_PATH)
    if not complete or not config.METRICS_PATH.exists():
        write_placeholder_metrics(config.ONNX_PATH, config.METRICS_PATH, config.DATA_DIR)
    return {"onnx": str(config.ONNX_PATH), "h5": str(config.H5_PATH) if config.H5_PATH.exists() else None}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keras", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(ensure_placeholder_files(parse_args().keras), indent=2))
