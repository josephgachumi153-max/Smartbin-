#!/usr/bin/env python
"""Train the SmartBin MobileNetV2 classifier from a RealWaste archive.

The Colab notebook expects eight classes. RealWaste also contains a
``Food Organics`` class, which is intentionally excluded here so the model
matches the SmartBin web application's taxonomy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
import zipfile
from pathlib import Path


CLASS_NAMES = [
    "cardboard",
    "paper",
    "textile_trash",
    "glass",
    "metal",
    "vegetation",
    "miscellaneous_trash",
    "plastic",
]

SOURCE_FOLDERS = {
    "cardboard": "Cardboard",
    "paper": "Paper",
    "textile_trash": "Textile Trash",
    "glass": "Glass",
    "metal": "Metal",
    "vegetation": "Vegetation",
    "miscellaneous_trash": "Miscellaneous Trash",
    "plastic": "Plastic",
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(r"C:\Users\steph\Downloads\archive.zip"),
        help="RealWaste archive to extract and train from.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("training_data"),
        help="Directory where the archive is extracted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model"),
        help="Directory for the trained model and metadata.",
    )
    parser.add_argument("--epochs-head", type=int, default=8)
    parser.add_argument("--epochs-fine", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def extract_archive(archive: Path, destination: Path) -> Path:
    if not archive.is_file():
        raise FileNotFoundError(f"Archive not found: {archive.resolve()}")

    expected = destination / "realwaste-main" / "RealWaste"
    if expected.is_dir():
        print(f"Using existing extracted dataset: {expected.resolve()}", flush=True)
        return expected

    destination.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive.resolve()} to {destination.resolve()} ...", flush=True)
    with zipfile.ZipFile(archive) as archive_file:
        archive_file.extractall(destination)
    if not expected.is_dir():
        raise FileNotFoundError(f"Expected dataset folder was not found: {expected.resolve()}")
    return expected


def image_records(dataset_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for class_name in CLASS_NAMES:
        class_dir = dataset_root / SOURCE_FOLDERS[class_name]
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir.resolve()}")
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            records.append(
                {
                    "path": str(path.resolve()),
                    "class_name": class_name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    if not records:
        raise ValueError(f"No supported image files found below {dataset_root.resolve()}")

    # Keep only one copy of exact duplicates and reject conflicting labels.
    by_hash: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_hash.setdefault(str(record["sha256"]), []).append(record)
    for digest, group in by_hash.items():
        if len({str(item["class_name"]) for item in group}) > 1:
            raise ValueError(f"Duplicate image has conflicting labels: {digest[:12]}")
    return [group[0] for group in by_hash.values()]


def stratified_split(
    records: list[dict[str, object]], seed: int
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Create deterministic 70/15/15 class-stratified splits."""
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, object]]] = {name: [] for name in CLASS_NAMES}
    for record in records:
        by_class[str(record["class_name"])].append(record)

    train: list[dict[str, object]] = []
    validation: list[dict[str, object]] = []
    test: list[dict[str, object]] = []
    for class_name in CLASS_NAMES:
        items = by_class[class_name]
        rng.shuffle(items)
        test_count = max(1, round(len(items) * 0.15))
        validation_count = max(1, round(len(items) * 0.15))
        test.extend(items[:test_count])
        validation.extend(items[test_count : test_count + validation_count])
        train.extend(items[test_count + validation_count :])
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "class_name", "sha256"])
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Import TensorFlow after argument parsing so an invalid path fails fast.
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = extract_archive(args.archive, args.dataset_root)
    records = image_records(dataset_root)
    train_records, validation_records, test_records = stratified_split(records, args.seed)
    write_csv(args.output_dir / "dataset_audit.csv", records)

    print(f"Classes: {CLASS_NAMES}", flush=True)
    print(f"Usable unique images: {len(records)}", flush=True)
    print(
        f"Splits: train={len(train_records)}, validation={len(validation_records)}, "
        f"test={len(test_records)}",
        flush=True,
    )
    for name, split in (
        ("train", train_records),
        ("validation", validation_records),
        ("test", test_records),
    ):
        counts = {class_name: 0 for class_name in CLASS_NAMES}
        for record in split:
            counts[str(record["class_name"])] += 1
        print(f"{name}: {counts}", flush=True)

    image_size = (160, 160)
    class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    autotune = tf.data.AUTOTUNE

    def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    def make_dataset(split: list[dict[str, object]], training: bool = False) -> tf.data.Dataset:
        paths = [str(record["path"]) for record in split]
        labels = [class_to_index[str(record["class_name"])] for record in split]
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            dataset = dataset.shuffle(len(split), seed=args.seed, reshuffle_each_iteration=True)
        return dataset.map(load_image, num_parallel_calls=autotune).batch(args.batch_size).prefetch(autotune)

    train_ds = make_dataset(train_records, training=True)
    validation_ds = make_dataset(validation_records)
    test_ds = make_dataset(test_records)

    print("Loading ImageNet MobileNetV2 weights...", flush=True)
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
        ],
        name="augmentation",
    )
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=image_size + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    inputs = keras.Input(shape=image_size + (3,))
    features = augmentation(inputs)
    features = base_model(features, training=False)
    features = layers.GlobalAveragePooling2D()(features)
    features = layers.Dropout(0.25)(features)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(features)
    model = keras.Model(inputs, outputs, name="smartbin2_mobilenetv2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    best_path = args.output_dir / "smartbin2_best.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6
        ),
    ]

    started_at = time.time()
    print("Training classification head...", flush=True)
    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=args.epochs_head,
        callbacks=callbacks,
    )

    # Continue from the best head checkpoint before fine-tuning.
    if best_path.is_file():
        model = keras.models.load_model(best_path)
        base_model = model.get_layer("mobilenetv2_1.00_160")

    base_model.trainable = True
    fine_tune_at = max(0, len(base_model.layers) - 30)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("Fine-tuning MobileNetV2...", flush=True)
    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=args.epochs_fine,
        callbacks=callbacks,
    )

    if best_path.is_file():
        model = keras.models.load_model(best_path)
    model.save(best_path)
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

    labels_path = args.output_dir / "class_labels.json"
    labels_path.write_text(json.dumps(CLASS_NAMES, indent=2) + "\n", encoding="utf-8")
    label_map_path = args.output_dir / "label_map.json"
    label_map_path.write_text(
        json.dumps({str(index): name for index, name in enumerate(CLASS_NAMES)}, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "architecture": "MobileNetV2 transfer learning",
        "image_size": list(image_size),
        "classes": CLASS_NAMES,
        "dataset": "RealWaste (Food Organics excluded)",
        "image_count": len(records),
        "train_count": len(train_records),
        "validation_count": len(validation_records),
        "test_count": len(test_records),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "seed": args.seed,
        "training_seconds": round(time.time() - started_at, 1),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Saved trained model to {best_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
