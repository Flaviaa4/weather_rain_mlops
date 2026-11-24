import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from src import config

LOGGER = logging.getLogger("onnx_export")


def input_signature(size=config.IMAGE_SIZE):
    import tensorflow as tf
    return (tf.TensorSpec((None, size, size, config.CHANNELS), tf.float32, name="image"),)


def export_with_from_keras(model, onnx_path):
    import tf2onnx
    model.output_names = ["probabilities"]
    tf2onnx.convert.from_keras(model, input_signature=input_signature(), output_path=str(onnx_path))


def export_with_saved_model(model, onnx_path):
    with tempfile.TemporaryDirectory() as folder:
        saved_dir = Path(folder) / "saved_model"
        model.export(str(saved_dir))
        command = [sys.executable, "-m", "tf2onnx.convert", "--saved-model", str(saved_dir), "--output", str(onnx_path)]
        subprocess.run(command, check=True, capture_output=True, text=True)


def convert(model, onnx_path):
    try:
        export_with_from_keras(model, onnx_path)
        return "from_keras"
    except Exception as error:
        LOGGER.warning("from_keras failed (%s), using the SavedModel CLI", error)
    export_with_saved_model(model, onnx_path)
    return "saved_model_cli"


def add_metadata(onnx_path, metadata):
    import onnx
    from onnx import helper
    model = onnx.load(str(onnx_path))
    del model.metadata_props[:]
    helper.set_model_props(model, {key: str(value) for key, value in metadata.items()})
    onnx.save(model, str(onnx_path))


def export_keras_to_onnx(model, onnx_path, metadata=None):
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    partial = onnx_path.with_name(onnx_path.name + ".partial")
    method = convert(model, partial)
    add_metadata(partial, {"export_method": method, **(metadata or {})})
    os.replace(partial, onnx_path)
    return method
