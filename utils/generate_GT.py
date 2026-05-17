import argparse
import os
from PIL import Image


def resize_images(source_dir, target_dir, target_size):
    os.makedirs(target_dir, exist_ok=True)
    for filename in os.listdir(source_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)
            img = Image.open(source_path).convert("RGB")
            img = img.resize(target_size, Image.BILINEAR)
            img.save(target_path)
    print("Image resizing completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Resize images from source directory into target directory."
    )
    parser.add_argument(
        "--source", required=True,
        help="Path to the source image directory"
    )
    parser.add_argument(
        "--target", required=True,
        help="Path to the output directory"
    )
    parser.add_argument(
        "--width", type=int, default=384,
        help="Target width in pixels (default: 384)"
    )
    parser.add_argument(
        "--height", type=int, default=512,
        help="Target height in pixels (default: 512)"
    )
    args = parser.parse_args()
    resize_images(args.source, args.target, (args.width, args.height))


if __name__ == "__main__":
    main()
