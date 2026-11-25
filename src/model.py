import time

import keras
import numpy as np
from keras import layers, regularizers

from src import config


def kernel_regularizer(l2):
    return regularizers.L2(l2) if l2 > 0 else None


def conv_unit(x, filters, cfg):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=not cfg["batch_norm"],
                      kernel_regularizer=kernel_regularizer(cfg["l2"]))(x)
    if cfg["batch_norm"]:
        x = layers.BatchNormalization(momentum=config.config_setting(cfg, "bn_momentum"))(x)
    return layers.Activation("relu")(x)


def conv_block(x, filters, cfg):
    for _ in range(cfg["convs_per_block"]):
        x = conv_unit(x, filters, cfg)
    return layers.MaxPooling2D()(x)


def pool_features(x, pooling):
    return layers.GlobalAveragePooling2D()(x) if pooling == "gap" else layers.Flatten()(x)


def classifier_head(x, cfg, n_classes):
    x = pool_features(x, cfg["pooling"])
    if cfg["dropout"] > 0:
        x = layers.Dropout(cfg["dropout"])(x)
    x = layers.Dense(cfg["dense_units"], activation="relu", kernel_regularizer=kernel_regularizer(cfg["l2"]))(x)
    if cfg["dropout"] > 0:
        x = layers.Dropout(cfg["dropout"])(x)
    return layers.Dense(n_classes, activation="softmax", name="probabilities")(x)


def model_input(cfg):
    return keras.Input(shape=(cfg["image_size"], cfg["image_size"], config.CHANNELS), name="image")


def build_cnn(cfg, n_classes=len(config.CLASS_NAMES)):
    inputs = model_input(cfg)
    x = inputs
    for block in range(cfg["blocks"]):
        x = conv_block(x, cfg["base_filters"] * 2 ** block, cfg)
    return keras.Model(inputs, classifier_head(x, cfg, n_classes), name=cfg["name"])


# the baseline settings (2 blocks of 1 conv, flatten, no extras) give the same network as the Section 3 baseline
def build_model(cfg):
    return build_cnn(cfg)


def build_optimizer(cfg):
    rate = cfg["learning_rate"]
    if cfg["optimizer"] == "adam":
        return keras.optimizers.Adam(rate)
    if cfg["optimizer"] == "adamw":
        return keras.optimizers.AdamW(rate, weight_decay=cfg["weight_decay"])
    if cfg["optimizer"] == "sgd":
        return keras.optimizers.SGD(rate, momentum=0.9, nesterov=True)
    if cfg["optimizer"] == "rmsprop":
        return keras.optimizers.RMSprop(rate)
    raise ValueError(f"unknown optimizer: {cfg['optimizer']}")


def compile_model(model, cfg):
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=cfg["label_smoothing"])
    model.compile(optimizer=build_optimizer(cfg), loss=loss, metrics=["accuracy"])
    return model


class EpochLogger(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.started = time.perf_counter()
        self.rate = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))

    def on_epoch_end(self, epoch, logs=None):
        logs["epoch_seconds"] = time.perf_counter() - self.started
        logs["learning_rate"] = self.rate


class RestoreBestWeights(keras.callbacks.Callback):
    def __init__(self, start_epoch=0):
        super().__init__()
        self.start_epoch = start_epoch

    def on_train_begin(self, logs=None):
        self.best_loss, self.best_weights = {"all": np.inf, "after_start": np.inf}, {}

    def remember(self, key, loss):
        if loss < self.best_loss[key]:
            self.best_loss[key], self.best_weights[key] = loss, self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        self.remember("all", logs["val_loss"])
        if epoch >= self.start_epoch:
            self.remember("after_start", logs["val_loss"])

    def on_train_end(self, logs=None):
        weights = self.best_weights.get("after_start", self.best_weights.get("all"))
        if weights is not None:
            self.model.set_weights(weights)


class LearningRateWarmup(keras.callbacks.Callback):
    def __init__(self, target_rate, epochs):
        super().__init__()
        self.target_rate, self.epochs = target_rate, epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.epochs:
            self.model.optimizer.learning_rate = self.target_rate * (epoch + 1) / self.epochs


def stopping_callbacks(cfg):
    start = config.config_setting(cfg, "min_epochs")
    return [RestoreBestWeights(start),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["early_stop_patience"], start_from_epoch=start)]


def schedule_callbacks(cfg):
    callbacks = []
    if config.config_setting(cfg, "warmup_epochs") > 0:
        callbacks.append(LearningRateWarmup(cfg["learning_rate"], config.config_setting(cfg, "warmup_epochs")))
    if cfg["lr_schedule"]:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=config.LR_FACTOR,
                                                           patience=config.LR_PATIENCE, min_lr=config.MIN_LR))
    return callbacks


def get_callbacks(cfg, history_path, backup_dir=None):
    callbacks = stopping_callbacks(cfg) + schedule_callbacks(cfg)
    if backup_dir is not None:
        callbacks.append(keras.callbacks.BackupAndRestore(str(backup_dir)))
    callbacks += [EpochLogger(), keras.callbacks.CSVLogger(str(history_path), append=backup_dir is not None)]
    return callbacks


# clone_model recompiles the clone in Keras 3 and saves a fresh optimizer of the same size, from_config does not
def uncompiled_copy(model):
    copy = keras.Model.from_config(model.get_config())
    copy.set_weights(model.get_weights())
    return copy


def save_without_optimizer(model, path):
    uncompiled_copy(model).save(path)
    return path


def batch_norm_layers(model):
    return [layer for layer in model.layers if isinstance(layer, layers.BatchNormalization)]


def set_batch_norm_momentum(model, momentum):
    for layer in batch_norm_layers(model):
        layer.momentum = momentum
    return model


def refresh_batch_norm_statistics(model, images):
    momenta = [layer.momentum for layer in batch_norm_layers(model)]
    set_batch_norm_momentum(model, 0.0)
    model(images, training=True)
    for layer, momentum in zip(batch_norm_layers(model), momenta):
        layer.momentum = momentum
    return len(momenta)


def compute_class_weights(labels, n_classes=len(config.CLASS_NAMES)):
    counts = np.bincount(labels, minlength=n_classes)
    return {k: float(len(labels) / (n_classes * counts[k])) for k in range(n_classes) if counts[k] > 0}


def train_model(model, train_dataset, val_dataset, cfg, callbacks, class_weight=None):
    return model.fit(train_dataset, validation_data=val_dataset, epochs=cfg["max_epochs"],
                     callbacks=callbacks, class_weight=class_weight, verbose=0)


def predict_probabilities(model, images, batch_size=config.BATCH_SIZE):
    return model.predict(images, batch_size=batch_size, verbose=0)


def evaluate_model(model, images, labels, batch_size=config.BATCH_SIZE):
    from src.metrics import classification_metrics
    probabilities = predict_probabilities(model, images, batch_size)
    return classification_metrics(probabilities, labels), probabilities
