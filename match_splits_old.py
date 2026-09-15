#!/usr/bin/env python3
"""
Recherche les patients communs à data/optic_disc/{train,val}/images.

Les originales sont recherchées dans pngs.txt ET pngs2.txt par défaut.
Le NAS doit être accessible pour comparer les octets, puis les pixels.
Le dHash optionnel fournit seulement des candidats à vérifier visuellement.
Les labels YOLO ne sont pas utilisés et le dataset n'est jamais modifié.

Usage depuis le dépôt :
    python match_splits.py
    python match_splits.py data/optic_disc pngs2.txt pngs.txt
    python match_splits.py --dhash-threshold 6

Sorties : mapping.csv (une ligne par candidat), patient_overlap.csv,
train.txt, val.txt, unmatched.txt, missing_originals.txt, unreadable_originals.txt.
L'identité patient est inférée du nom NAS : date et côté sont ignorés,
les codes BL/BL2/AS/apne sont traités comme des suffixes d'acquisition.
Les identifiants IREDO et les normalisations moins sûres restent à vérifier.
Des alias d'un même patient ne peuvent pas être déduits des images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "val")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PATIENT_PATTERN = re.compile(
    r"^(?P<patient>[A-Z]+(?:_?\d{3,})?)(?P<variant>[BH]?)(?=_+"
    r"(?:[LR]\d*|HD|BL\d*|AS|APNE|\d+)_)",
    re.IGNORECASE,
)

def file_md5(path: Path, chunk: int = 1 << 20) -> str | None:
    """md5 des octets bruts du fichier."""
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            while block := f.read(chunk):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def _imread(path: Path) -> np.ndarray | None:
    """Lecture robuste (gère les chemins non-ASCII sous Windows)."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def pixel_md5(path: Path) -> str | None:
    """md5 du tableau de pixels décodé : insensible au ré-encodage."""
    img = _imread(path)
    if img is None:
        return None
    header = f"{img.shape}:{img.dtype}:".encode("ascii")
    return hashlib.md5(header + np.ascontiguousarray(img).tobytes()).hexdigest()


def dhash(path: Path, size: int = 16) -> int | None:
    """
    Hash perceptuel (difference hash) sur size*size bits.
    Insensible au redimensionnement et à la compression légère.
    """
    img = _imread(path)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def patient_info(path: Path) -> tuple[str, str]:
    """Identifiant inféré et réserve éventuelle ; ne déduit pas d'alias certains."""
    name = PureWindowsPath(str(path)).stem.upper()
    name = re.sub(r"^\d{6}_", "", name)
    iredo = re.match(r"IREDO\s*-?\s*(\d+)(?=[\s_-])", name)
    if iredo:
        # Le numéro 40 apparaît avec HA et VS : ne pas certifier ces identités.
        return f"IREDO{iredo[1]}", "Identité IREDO à valider (numéros parfois ambigus)"
    note = ""
    if re.match(r"^\d{6}[A-Z]", name):
        name = name[6:]
        note = "Date accolée à l'identifiant retirée : alias à valider"
    if name.startswith("JUSTINEPRESSURE_"):
        return "JUSTINEPRESSURE", "Nom de protocole : identité patient à valider"
    match = PATIENT_PATTERN.match(name)
    if not match:
        return "", "Format d'identifiant non reconnu"
    if match["variant"]:
        note = "Suffixe b/h retiré : alias patient à valider"
    return match["patient"].replace("_", ""), note


def write_lines(path: Path, lines) -> None:
    lines = list(lines)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_original_paths(
    list_files: list[Path],
) -> tuple[list[Path], list[str], dict[Path, list[str]]]:
    paths, missing = [], []
    seen = set()
    sources = defaultdict(list)
    for list_file in list_files:
        with list_file.open(encoding="utf-8-sig") as stream:
            for line in stream:
                raw = line.strip().strip('"')
                if not raw:
                    continue
                path = Path(raw)
                sources[path].append(str(list_file))
                key = path
                if key in seen:
                    continue
                seen.add(key)
                if path.is_file():
                    paths.append(path)
                else:
                    missing.append(raw)
    return paths, missing, sources


def collect_dataset_images(dataset_dir: Path) -> list[tuple[Path, str]]:
    items = []
    for split in SPLITS:
        folder = dataset_dir / split / "images"
        if not folder.is_dir():
            raise ValueError(f"Dossier requis absent : {folder}")
        images = sorted(p for p in folder.rglob("*")
                        if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if not images:
            raise ValueError(f"Aucune image dans : {folder}")
        items.extend((path, split) for path in images)
    return items


def match_dataset(dataset_dir: Path, list_files: list[Path], out_dir: Path,
                  dhash_threshold: int = 0) -> None:
    dataset = collect_dataset_images(dataset_dir)
    originals, missing, sources = read_original_paths(list_files)
    if not originals:
        raise ValueError("Aucune originale accessible. Vérifier les listes et "
                         "la connexion au NAS (lecteur Y:). Aucun audit effectué.")
    # Empêche les rapports d'écraser les données ou les listes sources.
    output = out_dir.resolve()
    if output == dataset_dir.resolve() or dataset_dir.resolve() in output.parents:
        raise ValueError("Le dossier de sortie doit être en dehors du dataset.")
    report_names = {"mapping.csv", "patient_overlap.csv", "train.txt", "val.txt",
                    "unmatched.txt", "missing_originals.txt", "unreadable_originals.txt"}
    if any(p.resolve().parent == output and p.name in report_names for p in list_files):
        raise ValueError("Une liste source serait écrasée par un rapport.")

    print(f"{len(dataset)} images du dataset ; {len(originals)} originales accessibles ; "
          f"{len(missing)} chemins NAS absents.")
    idx_bytes, idx_pixels = defaultdict(list), defaultdict(list)
    perceptual, unreadable = [], []
    for n, path in enumerate(originals, 1):
        byte_key, pixel_key = file_md5(path), pixel_md5(path)
        if byte_key is not None:
            idx_bytes[byte_key].append(path)
        if pixel_key is not None:
            idx_pixels[pixel_key].append(path)
        if byte_key is None or pixel_key is None:
            unreadable.append(str(path))
        if dhash_threshold > 0:
            value = dhash(path)
            if value is not None:
                perceptual.append((value, path))
        if n % 200 == 0:
            print(f"Indexation : {n}/{len(originals)}", file=sys.stderr)

    rows, unmatched = [], []
    for img, split in dataset:
        candidates = idx_bytes.get(file_md5(img), [])
        method, distances = "md5_octets", {}
        if not candidates:
            candidates = idx_pixels.get(pixel_md5(img), [])
            method = "md5_pixels"
        if not candidates and dhash_threshold > 0:
            value = dhash(img)
            if value is not None:
                # Conserve tous les candidats sous le seuil, sans attribution forcée.
                distances = {p: hamming(value, h) for h, p in perceptual
                             if hamming(value, h) <= dhash_threshold}
                candidates = sorted(distances, key=lambda p: (distances[p], str(p)))
                method = "dhash"
        if not candidates:
            unmatched.append(f"{split}\t{img}")
            continue
        info = {p: patient_info(p) for p in candidates}
        identities = {identity for identity, _ in info.values()}
        # Plusieurs copies exactes du même patient restent exploitables.
        certain = (method != "dhash" and len(identities) == 1
                   and "" not in identities and not any(note for _, note in info.values()))
        for original in candidates:
            rows.append({
                "split": split, "image": str(img), "original": str(original),
                "source_lists": "\n".join(sorted(set(sources[original]))),
                "patient_id": info[original][0], "patient_note": info[original][1],
                "method": method,
                "distance": distances.get(original, ""),
                "candidate_count": len(candidates),
                "status": "exact_patient" if certain else "review",
            })

    by_patient = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["patient_id"]:
            by_patient[row["patient_id"]][row["split"]].append(row)
    overlaps = []
    for patient, groups in sorted(by_patient.items()):
        if not all(split in groups for split in SPLITS):
            continue
        confirmed = all(any(r["status"] == "exact_patient" for r in groups[s])
                        for s in SPLITS)
        overlaps.append({
            "patient_id": patient,
            "status": "exact_overlap" if confirmed else "possible_overlap",
            "train_images": "\n".join(sorted({r["image"] for r in groups["train"]})),
            "val_images": "\n".join(sorted({r["image"] for r in groups["val"]})),
            "train_originals": "\n".join(sorted({r["original"] for r in groups["train"]})),
            "val_originals": "\n".join(sorted({r["original"] for r in groups["val"]})),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "mapping.csv",
              ["split", "image", "original", "source_lists", "patient_id",
               "patient_note", "method", "distance",
               "candidate_count", "status"], rows)
    write_csv(out_dir / "patient_overlap.csv",
              ["patient_id", "status", "train_images", "val_images",
               "train_originals", "val_originals"], overlaps)
    for split in SPLITS:
        write_lines(out_dir / f"{split}.txt",
                    sorted({r["original"] for r in rows if r["split"] == split}))
    write_lines(out_dir / "unmatched.txt", unmatched)
    write_lines(out_dir / "missing_originals.txt", missing)
    write_lines(out_dir / "unreadable_originals.txt", unreadable)

    exact = sum(r["status"] == "exact_overlap" for r in overlaps)
    unresolved = {(r["split"], r["image"]) for r in rows if r["status"] == "review"}
    print(f"Patients communs (identifiants inférés des noms NAS) : "
          f"{exact} exacts, {len(overlaps) - exact} possibles.")
    print(f"Images sans correspondance : {len(unmatched)} ; "
          f"images à vérifier : {len(unresolved)}.")
    if missing or unreadable or unmatched or unresolved:
        print("[!] Audit incomplet : consulter les rapports avant toute conclusion.")
    print("L'absence de patient commun détecté ne garantit pas l'absence de fuite : "
          "les alias et identifiants NAS doivent être validés.")
    print(f"Rapports : {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", type=Path,
                        default=ROOT / "data" / "optic_disc")
    parser.add_argument("lists", nargs="*", type=Path,
                        default=[ROOT / "pngs.txt", ROOT / "pngs2.txt"],
                        help="listes NAS (défaut : pngs.txt et pngs2.txt)")
    parser.add_argument("-o", "--out", type=Path, default=ROOT / "splits_out")
    parser.add_argument("--dhash-threshold", type=int, default=0,
                        help="0 : désactivé (défaut) ; 1–256 : candidats perceptuels "
                             "à vérifier, par exemple 6.")
    args = parser.parse_args()
    if not 0 <= args.dhash_threshold <= 256:
        parser.error("--dhash-threshold doit être compris entre 0 et 256.")
    try:
        match_dataset(args.dataset, args.lists, args.out, args.dhash_threshold)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Erreur : {exc}\n")


if __name__ == "__main__":
    main()
