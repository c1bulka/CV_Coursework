import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from train import PixelShuffle, psnr_metric


MODEL_PATH = "runs_bsds/train_best.keras"
UPSCALE = 3
if len(sys.argv) < 2:
    print("Usage: python demo.py <image_path>")
    sys.exit(1)
IMAGE_PATH = sys.argv[1]

# model load

model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={"PixelShuffle": PixelShuffle, "psnr_metric": psnr_metric}
)


# utilities
def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype("float32") / 255.0


def downscale(img, factor=UPSCALE):
    h, w = img.shape[:2]
    lr = cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    return lr


def upscale(lr):
    sr = model.predict(lr[None, ...], verbose=0)[0]
    return np.clip(sr, 0, 1)


def demo_img():
    hr = load_img(IMAGE_PATH)
    lr = downscale(hr, UPSCALE)
    sr = upscale(lr)

    # bicubic resize for reference
    bic = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)

    # display all images
    fig, ax = plt.subplots(1, 4, figsize=(14, 5))
    ax[0].imshow(hr);  ax[0].set_title("Original")
    ax[1].imshow(lr);  ax[1].set_title(f"Downscaled x{UPSCALE}")
    ax[2].imshow(bic); ax[2].set_title("Bicubic Resize")
    ax[3].imshow(sr);  ax[3].set_title("Network Output")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_img()
