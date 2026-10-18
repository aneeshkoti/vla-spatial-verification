"""Copies the result tables of an earlier run into this repository's results/ folder (same v2 ... v7 layout).

    python scripts/collect_results.py --src ~/smolvla_project/results

Only the tables are copied (.csv, .json, .jsonl). Images, videos, saved scene images and model outputs are skipped, and so is any
file above 20 MB. A MANIFEST.json with the size and SHA-1 of every copied file is written next to them.
"""
import argparse
import hashlib
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDERS = ["v2", "v4", "v5", "v6", "v7"]                 # v3 holds results of an older version of the method (not used)
KEEP_EXT = (".csv", ".json", ".jsonl")
SKIP_DIRS = {"scenes", "frames", "qualitative", "demo", "detections", "figures", "paper_figures", "official_lerobot_eval",
             "official_xvla_eval", "rollouts", "images"}
MAX_MB = 20


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the results folder of the earlier run (it contains v4, v5, v6, v7)")
    ap.add_argument("--dst", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    src, dst = os.path.expanduser(args.src), os.path.expanduser(args.dst)
    if not os.path.isdir(src):
        raise SystemExit(f"folder not found: {src}")
    manifest, skipped_big = [], []
    for folder in FOLDERS:
        base = os.path.join(src, folder)
        if not os.path.isdir(base):
            print(f"  (no {folder} folder)")
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for fn in sorted(filenames):
                if not fn.endswith(KEEP_EXT):
                    continue
                if folder == "v2" and fn != "baseline_clean.csv":
                    continue                              # from v2 only the old baseline (used for a reproducibility check)
                p = os.path.join(dirpath, fn)
                size = os.path.getsize(p)
                rel = os.path.relpath(p, src)
                if size > MAX_MB * 1024 * 1024:
                    skipped_big.append((rel, size))
                    continue
                out = os.path.join(dst, rel)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                shutil.copy2(p, out)
                manifest.append({"file": rel.replace(os.sep, "/"), "bytes": size, "sha1": sha1(out)})
    with open(os.path.join(dst, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    total = sum(m["bytes"] for m in manifest)
    print(f"copied {len(manifest)} files ({total / 1e6:.1f} MB) into {dst}")
    for folder in FOLDERS:
        n = [m for m in manifest if m["file"].startswith(folder + "/")]
        if n:
            print(f"   {folder}: {len(n):3d} files, {sum(m['bytes'] for m in n) / 1e6:6.2f} MB")
    for rel, size in skipped_big:
        print(f"   skipped (larger than {MAX_MB} MB): {rel} ({size / 1e6:.0f} MB)")
    print("next: python scripts/reproduce_numbers.py")


if __name__ == "__main__":
    main()
