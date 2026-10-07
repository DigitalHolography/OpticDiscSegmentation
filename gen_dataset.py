"""Copie les images des deux CSV dans dataset/train et dataset/test.

Usage : python gen_dataset.py
        python gen_dataset.py --train chemin/train.csv --test chemin/test.csv --output dataset
"""

import argparse
import csv
import shutil
from pathlib import Path


def main():
    downloads = Path.home() / "Downloads"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=downloads / "sources_kfold_train.csv")
    parser.add_argument("--test", type=Path, default=downloads / "sources_kfold_test.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "dataset")
    args = parser.parse_args()
    errors = 0

    for split, csv_path in (("train", args.train), ("test", args.test)):
        destination = args.output / split
        destination.mkdir(parents=True, exist_ok=True)
        copied = 0
        with csv_path.open(newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                source = Path(row["SOURCE"].strip().strip('"'))
                name = row["NAME"].strip()
                if not name or Path(name).name != name:
                    raise ValueError(f"Nom de fichier invalide : {name!r}")
                try:
                    shutil.copy2(source, destination / name)
                    copied += 1
                except OSError as error:
                    print(f"Échec : {source} -> {name} : {error}")
                    errors += 1
        print(f"{split} : {copied} images copiées dans {destination}")

    if errors:
        raise SystemExit(f"{errors} image(s) non copiée(s).")


if __name__ == "__main__":
    main()
