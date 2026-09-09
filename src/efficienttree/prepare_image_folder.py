from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"
}


def parse_image_path(raw_line: str) -> Path | None:
    line = raw_line.strip().strip('"')
    if not line:
        return None

    parts = line.rsplit(maxsplit=1)

    # EfficientTree split format:
    # image_path label_path
    if len(parts) == 2 and Path(parts[1].strip('"')).suffix.lower() == ".txt":
        return Path(parts[0].strip('"'))

    # Image-only list
    return Path(line)


def clear_output_folder(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)

    for item in folder.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "linked"
    except OSError:
        shutil.copy2(source, destination)
        return "copied"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an image-only test folder from an EfficientTree split file."
    )
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.test_list.is_file():
        raise FileNotFoundError(f"Test list not found: {args.test_list}")

    clear_output_folder(args.output_dir)

    image_paths: list[Path] = []
    for raw_line in args.test_list.read_text(
        encoding="utf-8-sig"
    ).splitlines():
        image_path = parse_image_path(raw_line)
        if image_path is not None:
            image_paths.append(image_path)

    if not image_paths:
        raise RuntimeError("No image paths were found in the test list.")

    seen_names: set[str] = set()
    linked = 0
    copied = 0

    for image_path in image_paths:
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image file: {image_path}")

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if image_path.name.lower() in seen_names:
            raise RuntimeError(
                f"Duplicate image filename in test set: {image_path.name}"
            )
        seen_names.add(image_path.name.lower())

        destination = args.output_dir / image_path.name
        result = link_or_copy(image_path, destination)

        if result == "linked":
            linked += 1
        else:
            copied += 1

    print("=" * 72)
    print("TEST IMAGE FOLDER READY")
    print("=" * 72)
    print(f"Images: {len(image_paths)}")
    print(f"Hard-linked: {linked}")
    print(f"Copied: {copied}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
