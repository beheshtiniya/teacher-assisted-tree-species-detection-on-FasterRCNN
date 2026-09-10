from __future__ import annotations

import argparse
import py_compile
import shutil
from datetime import datetime
from pathlib import Path


MARKER = "# PATCH: validation EarlyStopping patience=7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely inject validation-based EarlyStopping(patience=7) into "
            "EfficientTree trainer/trainer.py."
        )
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(
            r"E:\FASTRCNN\FASTRCNN\EfficientTree-paper\EfficientTree-master"
        ),
    )
    parser.add_argument("--patience", type=int, default=7)
    return parser.parse_args()


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{description}: expected exactly one match, but found {count}. "
            "No source file was changed."
        )
    return text.replace(old, new, 1)


def main() -> None:
    args = parse_args()
    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")

    repo = args.repo.resolve()
    trainer = repo / "trainer" / "trainer.py"

    if not trainer.is_file():
        raise FileNotFoundError(f"trainer.py not found: {trainer}")

    original = trainer.read_text(encoding="utf-8")

    if MARKER in original:
        print("Early-stopping patch is already present; no changes were made.")
        print(f"Trainer: {trainer}")
        return

    patched = original

    # 1) Initialize a stopper once at the beginning of train().
    init_old = "        self.best_fitness = 0\n"
    init_new = (
        "        self.best_fitness = 0\n"
        f"        {MARKER}\n"
        "        from utils.torch_utils import EarlyStopping\n"
        f"        self.stopper = EarlyStopping(patience={args.patience})\n"
        "        self.current_fitness = None\n"
    )
    patched = replace_once(
        patched,
        init_old,
        init_new,
        "Could not locate the train() best_fitness initialization",
    )

    # 2) Save the current validation fitness separately from best_fitness.
    fitness_old = (
        "            fi = fitness(np.array(self.results).reshape(1, -1))"
    )
    fitness_new = (
        "            fi = fitness(np.array(self.results).reshape(1, -1))"
        "\n            self.current_fitness = "
        "float(np.asarray(fi).reshape(-1)[0])"
    )
    patched = replace_once(
        patched,
        fitness_old,
        fitness_new,
        "Could not locate the validation fitness calculation",
    )

    # 3) Stop only after validation, logging, and checkpoint saving have completed.
    epoch_old = "            self.after_epoch(callbacks, val)\n"
    epoch_new = (
        "            self.current_fitness = None\n"
        "            self.after_epoch(callbacks, val)\n"
        "            if (self.current_fitness is not None and\n"
        "                    self.stopper(self.epoch, self.current_fitness)):\n"
        "                LOGGER.info(\n"
        "                    f'Early stopping at epoch {self.epoch + 1}; '\n"
        "                    f'best validation epoch was '\n"
        "                    f'{self.stopper.best_epoch + 1}.')\n"
        "                break\n"
    )
    patched = replace_once(
        patched,
        epoch_old,
        epoch_new,
        "Could not locate the after_epoch() call in train()",
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = trainer.with_name(f"trainer.py.before_earlystop_{stamp}.bak")
    temporary = trainer.with_name("trainer.py.earlystop_candidate")

    shutil.copy2(trainer, backup)
    temporary.write_text(patched, encoding="utf-8")

    try:
        py_compile.compile(str(temporary), doraise=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Patched source did not compile. Original trainer.py is untouched. "
            f"Backup: {backup}"
        )

    trainer.write_text(patched, encoding="utf-8")
    temporary.unlink(missing_ok=True)

    # Compile the final filename as a second verification.
    py_compile.compile(str(trainer), doraise=True)

    print("=" * 82)
    print("EFFICIENTTREE EARLY STOPPING INJECTION COMPLETE")
    print("=" * 82)
    print(f"Trainer : {trainer}")
    print(f"Backup  : {backup}")
    print(f"Patience: {args.patience}")
    print("Metric  : validation fitness computed by the repository")
    print("Order   : validate -> save last/best -> early-stop decision")
    print("\nVerification command:")
    print(
        r'  Select-String -Path ".\trainer\trainer.py" '
        r'-Pattern "PATCH: validation EarlyStopping|self.stopper|current_fitness" '
        r'-Context 2,4'
    )


if __name__ == "__main__":
    main()
