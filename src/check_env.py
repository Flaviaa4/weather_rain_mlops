import importlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PROFILES = {
    "trainer": ["tensorflow", "keras", "tf2onnx", "onnx", "onnxruntime", "numpy", "pandas", "PIL", "matplotlib", "protobuf"],
    "all": ["tensorflow", "keras", "tf2onnx", "onnx", "onnxruntime", "numpy", "pandas", "PIL",
            "matplotlib", "fastapi", "uvicorn", "locust", "streamlit", "protobuf"],
}
MODULE_NAMES = {"protobuf": "google.protobuf"}


def import_version(package):
    try:
        module = importlib.import_module(MODULE_NAMES.get(package, package))
        return getattr(module, "__version__", "unknown")
    except Exception as error:
        return f"IMPORT FAILED: {type(error).__name__}: {error}"


def collect_versions(packages):
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    versions.update({package: import_version(package) for package in packages})
    return versions


def list_gpus():
    try:
        import tensorflow as tf
        return [device.name for device in tf.config.list_physical_devices("GPU")]
    except Exception as error:
        return [f"unavailable: {error}"]


def build_tiny_model():
    import keras
    inputs = keras.Input(shape=(32, 32, 3), name="image")
    x = keras.layers.Conv2D(4, 3, activation="relu")(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(4, activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs)


def export_with_from_keras(model, onnx_path):
    import tensorflow as tf
    import tf2onnx
    model.output_names = ["probabilities"]
    signature = (tf.TensorSpec((None, 32, 32, 3), tf.float32, name="image"),)
    tf2onnx.convert.from_keras(model, input_signature=signature, output_path=str(onnx_path))


def export_with_saved_model(model, folder, onnx_path):
    saved_dir = folder / "saved_model"
    model.export(str(saved_dir))
    command = [sys.executable, "-m", "tf2onnx.convert", "--saved-model", str(saved_dir), "--output", str(onnx_path)]
    subprocess.run(command, check=True, capture_output=True, text=True)


def try_export(model, folder):
    onnx_path = folder / "tiny.onnx"
    try:
        export_with_from_keras(model, onnx_path)
        return onnx_path, "from_keras"
    except Exception as error:
        print("from_keras failed:", type(error).__name__, str(error)[:300])
    export_with_saved_model(model, folder, onnx_path)
    return onnx_path, "saved_model_cli"


def onnx_predict(onnx_path, batch):
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: batch})[0]


def check_onnx_roundtrip():
    model = build_tiny_model()
    batch = np.random.default_rng(0).random((4, 32, 32, 3), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path, method = try_export(model, Path(tmp))
        keras_out = model.predict(batch, verbose=0)
        onnx_out = onnx_predict(onnx_path, batch)
    diff = float(np.abs(keras_out - onnx_out).max())
    return {"method": method, "max_abs_diff": diff, "passed": diff < 1e-4}


def check_h5_save():
    model = build_tiny_model()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tiny.h5"
        model.save(path)
        return path.stat().st_size > 0


def run_checks(include_tf, profile):
    report = {"versions": collect_versions(PROFILES[profile])}
    if include_tf:
        report["gpus"] = list_gpus()
        report["h5_save"] = check_h5_save()
        report["onnx_roundtrip"] = check_onnx_roundtrip()
    return report


def parse_profile(argv):
    return argv[argv.index("--profile") + 1] if "--profile" in argv else "all"


if __name__ == "__main__":
    result = run_checks(include_tf="--no-tf" not in sys.argv, profile=parse_profile(sys.argv))
    print(json.dumps(result, indent=2))
    failed = [k for k, v in result["versions"].items() if str(v).startswith("IMPORT FAILED")]
    print("FAILED IMPORTS:", failed if failed else "none")
    if "gpus" in result and not any(name.startswith("/physical_device:GPU") for name in result["gpus"]):
        print("WARNING: no GPU visible to TensorFlow. In Colab use Runtime > Change runtime type > T4 GPU.")
