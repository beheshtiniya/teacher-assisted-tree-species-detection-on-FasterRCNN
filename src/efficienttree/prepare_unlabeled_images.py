from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"
}


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


def parse_list_value(raw_line: str) -> str | None:
    line = raw_line.strip().strip('"')
    if not line:
        return None

    parts = line.rsplit(maxsplit=1)

    # EfficientTree format: image_path label_path
    if (
        len(parts) == 2
        and Path(parts[1].strip().strip('"')).suffix.lower() == ".txt"
    ):
        return parts[0].strip().strip('"')

    return line


def build_recursive_index(
    root: Path,
) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    if not root.is_dir():
        raise NotADirectoryError(f"Image root not found: {root}")

    by_name: dict[str, Path] = {}
    duplicate_names: dict[str, list[Path]] = {}

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        key = path.name.lower()

        if key in by_name:
            duplicate_names.setdefault(
                key,
                [by_name[key]],
            ).append(path)
        else:
            by_name[key] = path

    return by_name, duplicate_names


def resolve_image(
    value: str,
    images_root: Path,
    index: dict[str, Path],
    duplicates: dict[str, list[Path]],
) -> Path:
    candidate = Path(value)

    # Absolute/full path in the list.
    if candidate.is_file():
        return candidate

    # Relative path under the configured 640 image root.
    relative_candidate = images_root / candidate
    if relative_candidate.is_file():
        return relative_candidate

    # Filename-only lookup.
    key = candidate.name.lower()

    if key in duplicates:
        choices = "\n".join(
            f"  {path}"
            for path in duplicates[key]
        )
        raise RuntimeError(
            f"Ambiguous unlabeled filename: {candidate.name}\n{choices}"
        )

    if key in index:
        return index[key]

    raise FileNotFoundError(
        f"Unlabeled image could not be resolved: {value}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create an image-only 640x640 folder for the unlabeled list. "
            "The list may contain full paths, relative paths, or filenames."
        )
    )
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.list.is_file():
        raise FileNotFoundError(
            f"Unlabeled list not found: {args.list}"
        )

    values = [
        parsed
        for raw_line in args.list.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if (parsed := parse_list_value(raw_line)) is not None
    ]

    if not values:
        raise RuntimeError(
            f"No entries found in: {args.list}"
        )

    clear_output_folder(args.output_dir)

    index, duplicates = build_recursive_index(args.images_root)

    seen_names: set[str] = set()
    linked = 0
    copied = 0

    for value in values:
        source = resolve_image(
            value=value,
            images_root=args.images_root,
            index=index,
            duplicates=duplicates,
        )

        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(
                f"Unsupported image format: {source}"
            )

        name_key = source.name.lower()
        if name_key in seen_names:
            raise RuntimeError(
                f"Duplicate unlabeled filename: {source.name}"
            )
        seen_names.add(name_key)

        destination = args.output_dir / source.name
        result = link_or_copy(source, destination)

        if result == "linked":
            linked += 1
        else:
            copied += 1

    print("=" * 72)
    print("UNLABELED IMAGE FOLDER READY")
    print("=" * 72)
    print(f"List entries: {len(values)}")
    print(f"Prepared images: {len(seen_names)}")
    print(f"Hard-linked: {linked}")
    print(f"Copied: {copied}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
