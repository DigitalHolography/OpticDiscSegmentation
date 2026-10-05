#!/usr/bin/env python3
"""Inventorie les nouvelles prises M0 du NAS, sans copier ni modifier les images.

Depuis le dépôt, même installé sur un autre disque que le NAS :
    python inventory_nas.py
    python inventory_nas.py --nas-root Z:\\ --reference-root Y:\\
    python inventory_nas.py --references pngs.txt pngs2.txt

Par défaut : référence pngs.txt à côté de ce script, parcours intégral de Y:\\,
rapports dans nas_inventory/<horodatage> à côté du script.

Chaque chemin .../<acquisition>/png/<image>_M0.png de la référence exclut
l'acquisition entière et ses descendants, PAS son dossier de date ni le patient.
Les prises candidates sont les fichiers *_M0.png directement dans un dossier
nommé png, comme dans pngs.txt. Les autres types d'images sont ignorés.

L'identité est inférée du nom (trois lettres, chiffres ignorés). Les noms atypiques,
IREDO, protocoles et suffixes ambigus restent à vérifier. Aucune preuve d'identité
clinique ni de nouveauté de contenu : un fichier copié ailleurs peut réapparaître.
Liens, jonctions et points de réanalyse ne sont pas suivis pour éviter les boucles.
Python >= 3.10, bibliothèque standard seulement. Aucun accès NAS à l'import.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import stat
import sys

REPO = Path(__file__).resolve().parent


def path_key(path) -> str:
    """Comparaison Windows insensible à la casse, sans résoudre les chemins NAS."""
    return str(PureWindowsPath(str(path))).casefold().rstrip("\\")


def patient_info(filename: str) -> tuple[str, str, str, str]:
    """Retourne (code candidat, statut, identifiant brut, réserve).

    Une identité incertaine n'entre pas dans les effectifs patients confirmés.
    Les chiffres de prise ne sont pas absorbés dans le code patient.
    """
    name = PureWindowsPath(filename).stem.upper()
    name = re.sub(r"^\d{6}_", "", name)
    if name.startswith("IREDO"):
        token = re.match(r"IREDO\s*-?\s*\d+", name)
        return "", "review", token[0] if token else name, "Identifiant IREDO à résoudre"
    if "PRESSURE" in name:
        return "", "review", name.split("_")[0], "Nom de protocole : identité à résoudre"
    notes = []
    if re.match(r"^\d{6}[A-Z]", name):
        name = name[6:]
        notes.append("Date accolée retirée : alias à vérifier")
    match = re.match(
        r"^(?P<id>[A-Z]+(?:_?\d{3,})?)(?P<variant>[BH]?)(?=_+"
        r"(?:[LR]\d*|HD|BL\d*|AS|APNE|\d+)_)", name)
    if not match:
        return "", "review", name.split("_")[0], "Convention de nom non reconnue"
    raw = match["id"]
    code = re.fullmatch(r"([A-Z]{3})(?:_?\d+)?", raw)
    if not code:
        return "", "review", raw, "Identifiant différent du code à trois lettres"
    if match["variant"]:
        notes.append("Suffixe b/h retiré : alias à vérifier")
    return code[1], "review" if notes else "recognized", raw, " ; ".join(notes)


def read_reference_lists(list_files, reference_root, nas_root):
    """Parse les listes locales et traduit leur racine vers le montage à scanner.

    Refuse une ligne dont l'acquisition ne peut pas être déterminée plutôt que
    risquer d'exclure un niveau de dossier incorrect. Ne teste pas l'existence NAS.
    """
    reference_root = PureWindowsPath(str(reference_root))
    if not reference_root.is_absolute():
        raise ValueError("--reference-root doit être une racine Windows absolue")
    excluded, rows, fingerprints, seen = set(), [], {}, set()
    known_patients, uncertain_patients = set(), set()
    for filename in list_files:
        content = Path(filename).read_bytes()
        fingerprints[str(Path(filename).resolve())] = hashlib.sha256(content).hexdigest()
        for number, line in enumerate(content.decode("utf-8-sig").splitlines(), 1):
            raw = line.strip().strip('"')
            if not raw:
                continue
            path = PureWindowsPath(raw)
            if (not path.is_absolute() or ".." in path.parts or path.parent.name.lower() != "png"
                    or path.suffix.lower() != ".png"):
                raise ValueError(f"Référence non conforme {filename}:{number} : {raw}")
            acquisition = path.parent.parent
            try:
                relative = acquisition.relative_to(reference_root)
            except ValueError as exc:
                raise ValueError(f"Référence hors de {reference_root} : {raw}") from exc
            if not relative.parts:
                raise ValueError(f"Refus d'exclure la racine entière : {raw}")
            mapped = nas_root.joinpath(*relative.parts)
            excluded.add(path_key(mapped))
            code, status, _, _ = patient_info(path.name)
            if code:
                (known_patients if status == "recognized" else uncertain_patients).add(code)
            if path_key(path) in seen:
                continue
            seen.add(path_key(path))
            rows.append({"reference_image": raw, "excluded_acquisition": str(mapped),
                         "reference_list": str(filename)})
    if not rows:
        raise ValueError("Aucune référence ; arrêt pour éviter un inventaire sans exclusions")
    return excluded, rows, known_patients, uncertain_patients, fingerprints


def is_reparse(path: Path) -> bool:
    attributes = path.lstat()
    return path.is_symlink() or bool(getattr(attributes, "st_file_attributes", 0)
                                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def scan_nas(nas_root, excluded, known_patients, uncertain_patients):
    """Parcours complet hors exclusions ; conserve les erreurs d'accès au rapport."""
    rows, issues, skipped = [], [], []
    stats = {"visited_directories": 0, "excluded_directories": 0, "ignored_files": 0}

    def onerror(exc):
        issues.append({"path": str(exc.filename or ""), "error": str(exc)})

    for current, directories, files in os.walk(nas_root, topdown=True, followlinks=False, onerror=onerror):
        directory = Path(current)
        stats["visited_directories"] += 1
        directories.sort(key=str.casefold)
        retained = []
        for name in directories:
            child = directory / name
            if path_key(child) in excluded:
                stats["excluded_directories"] += 1
                continue
            try:
                if is_reparse(child):
                    skipped.append({"path": str(child), "reason": "Lien/jonction/point de réanalyse"})
                    continue
            except OSError as exc:
                onerror(exc)
                continue
            retained.append(name)
        directories[:] = retained
        if stats["visited_directories"] % 1000 == 0:
            print(f"{stats['visited_directories']} dossiers parcourus ; {len(rows)} images candidates",
                  file=sys.stderr)
        for name in sorted(files, key=str.casefold):
            if directory.name.lower() != "png" or not name.lower().endswith("_m0.png"):
                stats["ignored_files"] += 1
                continue
            image = directory / name
            try:
                if is_reparse(image):
                    skipped.append({"path": str(image), "reason": "Fichier lié/point de réanalyse"})
                    continue
            except OSError as exc:
                onerror(exc)
                continue
            code, status, raw, note = patient_info(name)
            if status != "recognized":
                reference_status = "identity_to_review"
            elif code in known_patients:
                reference_status = "present_in_reference"
            elif code in uncertain_patients:
                reference_status = "possible_in_reference"
            else:
                reference_status = "not_identified_in_reference"
            rows.append({"patient_code": code, "identity_status": status,
                         "reference_status": reference_status, "raw_patient_id": raw,
                         "nas_path": str(image), "acquisition_dir": str(directory.parent),
                         "identity_note": note})
    return rows, issues, skipped, stats


def summarize_patients(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["identity_status"] == "recognized":
            groups[row["patient_code"]].append(row)
    result = []
    for code, images in groups.items():
        result.append({"patient_code": code, "n_images": len(images),
                       "n_acquisitions": len({path_key(r["acquisition_dir"]) for r in images}),
                       "reference_status": images[0]["reference_status"],
                       "raw_patient_ids": json.dumps(sorted({r["raw_patient_id"] for r in images})),
                       "nas_paths": json.dumps([r["nas_path"] for r in images], ensure_ascii=False)})
    return sorted(result, key=lambda row: (-row["n_acquisitions"], row["patient_code"]))


def write_csv(path, fields, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nas-root", type=Path, default=Path("Y:/"), help="Racine du NAS à parcourir")
    parser.add_argument("--reference-root", default="Y:/", help="Racine dans les listes, à traduire vers --nas-root")
    parser.add_argument("--references", nargs="+", type=Path, default=[REPO / "pngs.txt"],
                        help="Listes locales des acquisitions déjà traitées (défaut : pngs.txt)")
    parser.add_argument("--out", type=Path, default=REPO / "nas_inventory",
                        help="Dossier parent local des rapports ; crée un sous-dossier neuf")
    args = parser.parse_args()
    try:
        if not args.nas_root.is_absolute():
            raise ValueError("--nas-root doit être absolu, par exemple Y:\\")
        root = args.nas_root.resolve()
        output_parent = args.out.resolve()
        if not root.is_dir():
            raise ValueError(f"NAS non accessible : {root}")
        if output_parent == root or output_parent.is_relative_to(root):
            raise ValueError("Les rapports doivent être écrits hors du NAS scanné")
        excluded, reference_rows, known, uncertain, hashes = read_reference_lists(
            args.references, args.reference_root, root)
        print(f"Parcours de {root} ; {len(excluded)} dossiers d'acquisition exclus.")
        rows, errors, skipped, stats = scan_nas(root, excluded, known, uncertain)
        patients = summarize_patients(rows)
        review = [r for r in rows if r["identity_status"] != "recognized"]
        output = output_parent / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True, exist_ok=False)
        fields = ["patient_code", "identity_status", "reference_status", "raw_patient_id",
                  "nas_path", "acquisition_dir", "identity_note"]
        write_csv(output / "new_images.csv", fields, rows)
        write_csv(output / "identity_review.csv", fields, review)
        write_csv(output / "patients.csv", ["patient_code", "n_images", "n_acquisitions",
                  "reference_status", "raw_patient_ids", "nas_paths"], patients)
        write_csv(output / "excluded_references.csv", ["reference_image", "excluded_acquisition", "reference_list"], reference_rows)
        write_csv(output / "scan_errors.csv", ["path", "error"], errors)
        write_csv(output / "skipped_links.csv", ["path", "reason"], skipped)
        with (output / "new_paths.txt").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(row["nas_path"] + "\n")
        summary = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                   "nas_root": str(root), "reference_root": args.reference_root,
                   "reference_sha256": hashes, "excluded_acquisitions": len(excluded),
                   "new_candidate_images": len(rows), "recognized_patient_codes": len(patients),
                   "identity_review_images": len(review), "access_errors": len(errors),
                   "skipped_links": len(skipped), "scan_incomplete": bool(errors or skipped), **stats}
        with (output / "summary.json").open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
        (output / "README.txt").write_text(
            "patients.csv : codes reconnus, triés par nombre de nouvelles acquisitions décroissant.\n"
            "n_images compte les fichiers M0 ; n_acquisitions compte leurs dossiers parents distincts.\n"
            "new_images.csv relie chaque image au patient et à l'acquisition.\n"
            "identity_review.csv contient les identités à vérifier ; elles ne comptent pas dans patients.csv.\n"
            "present_in_reference = code déjà reconnu dans la liste ; not_identified_in_reference =\n"
            "code non reconnu dans la liste, pas une preuve qu'il s'agit d'un nouveau patient réel.\n"
            "Les exclusions portent sur les chemins d'acquisition, pas sur le contenu des images.\n"
            "Les variantes ambiguës peuvent cacher des patients communs. Aucun label n'est généré.\n"
            "Les erreurs et liens ignorés sont listés ; le scan est alors incomplet.\n"
            "new_paths.txt contient tous les candidats, y compris les identités à vérifier.\n",
            encoding="utf-8")
        print(f"{len(rows)} images candidates ; {len(patients)} codes patients reconnus ; "
              f"{len(review)} images à identifier.\nRapports : {output}")
        if errors or skipped:
            print(f"Attention : inventaire incomplet ({len(errors)} erreurs, {len(skipped)} liens ignorés).")
        if errors:
            return 2
        return 0
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(1, f"Erreur : {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
