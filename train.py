import os
import tarfile
import random
import urllib.request
from glob import glob

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PixelShuffle(layers.Layer):
    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.scale)

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config

# configuration

DATA_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
DATA_ROOT = "./datasets/BSDS500"
UPSCALING_FACTOR = 3
CROP_HR = 192
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-3

OUT_DIR = "./runs_bsds"
os.makedirs(OUT_DIR, exist_ok=True)


# dataset download

def download_bsds500():
    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT, exist_ok=True)

    archive_path = os.path.join(DATA_ROOT, "bsds500.tgz")

    if not os.path.exists(archive_path):
        urllib.request.urlretrieve(DATA_URL, archive_path)

    extracted_flag = os.path.join(DATA_ROOT, "BSR")
    if not os.path.exists(extracted_flag):
        with tarfile.open(archive_path) as tar:
            tar.extractall(DATA_ROOT)


# dataset pipline

def load_image_paths():
    img_dir = os.path.join(DATA_ROOT, "BSR", "BSDS500", "data", "images")
    train = glob(os.path.join(img_dir, "train", "*.jpg"))
    val   = glob(os.path.join(img_dir, "test", "*.jpg"))   # use test as validation
    return train, val



def random_crop(img, size=CROP_HR):
    h, w, _ = img.shape
    if h < size or w < size:
        img = tf.image.resize(img, [size + 10, size + 10])
        h, w = img.shape[:2]
    top = random.randint(0, h - size)
    left = random.randint(0, w - size)
    return img[top:top+size, left:left+size, :]


def make_dataset(paths):
    def gen():
        for p in paths:
            img = Image.open(p).convert("RGB")
            img = np.array(img).astype("float32") / 255.0
            img = random_crop(img)
            yield img

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=tf.TensorSpec(shape=(CROP_HR, CROP_HR, 3), dtype=tf.float32)
    )

    ds = ds.map(lambda x: rgb_pair(x))

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def rgb_pair(hr):
    lr = tf.image.resize(hr, [CROP_HR // UPSCALING_FACTOR, CROP_HR // UPSCALING_FACTOR],
                         method=tf.image.ResizeMethod.AREA)
    return lr, hr


# model build


def build():
    conv_args = dict(activation="relu", kernel_initializer="orthogonal", padding="same")

    inp = keras.Input(shape=(None, None, 3))
    x = layers.Conv2D(64, 5, **conv_args)(inp)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(32, 3, **conv_args)(x)
    x = layers.Conv2D(3 * (UPSCALING_FACTOR ** 2), 3, padding="same")(x)

    # use wrapped pixel shuffle layer
    out = PixelShuffle(UPSCALING_FACTOR)(x)

    return keras.Model(inp, out, name=f"ESPCN_x{UPSCALING_FACTOR}")


# model training


def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


def train():
    download_bsds500()
    train_paths, val_paths = load_image_paths()

    train_ds = make_dataset(train_paths)
    val_ds = make_dataset(val_paths)

    model = build()
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss="mse",
        metrics=[psnr_metric]
    )
    model.summary()

    cb = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(OUT_DIR, "train_best.keras"),
            monitor="val_psnr_metric",
            mode="max",
            save_best_only=True
        ),
        keras.callbacks.CSVLogger(os.path.join(OUT_DIR, "train_log.csv"))
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=cb
    )

    model.save(os.path.join(OUT_DIR, "train_final.keras"))


if __name__ == "__main__":
    train()

