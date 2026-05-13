# download_model.py — Downloads VGG16 Caffe model files into models/
import os
import urllib.request

os.makedirs("models", exist_ok=True)

PROTOTXT_URL = (
    "https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8"
    "/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/"
    "VGG_ILSVRC_16_layers_deploy.prototxt"
)
CAFFEMODEL_URL = (
    "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/"
    "VGG_ILSVRC_16_layers.caffemodel"
)

PROTOTXT_PATH  = "models/VGG_ILSVRC_16_layers_deploy.prototxt"
CAFFEMODEL_PATH = "models/VGG_ILSVRC_16_layers.caffemodel"


def _progress(block, block_size, total):
    done = block * block_size
    pct  = min(done * 100 / total, 100) if total > 0 else 0
    mb   = done / (1024 * 1024)
    print(f"\r  Downloading... {pct:.1f}%  ({mb:.1f} MB)", end="", flush=True)


def download_file(url, dest):
    if os.path.exists(dest):
        print(f"[SKIP] {dest} already exists.")
        return
    print(f"[DOWNLOAD] {os.path.basename(dest)}")
    print(f"  Source : {url}")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\n  Saved  : {dest}  ({os.path.getsize(dest)/1e6:.1f} MB)")


if __name__ == "__main__":
    print("=" * 60)
    print("VGG16 Caffe Model Downloader")
    print("=" * 60)

    download_file(PROTOTXT_URL,    PROTOTXT_PATH)
    download_file(CAFFEMODEL_URL,  CAFFEMODEL_PATH)

    if os.path.exists(PROTOTXT_PATH) and os.path.exists(CAFFEMODEL_PATH):
        print("\n✓ Both model files are ready in models/")
    else:
        print("\n✗ Download incomplete. Check internet connection.")
        print("  Manual download:")
        print(f"  prototxt  → {PROTOTXT_URL}")
        print(f"  caffemodel → {CAFFEMODEL_URL}")
