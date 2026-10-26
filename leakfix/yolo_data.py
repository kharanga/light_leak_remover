import argparse, csv, glob, os, random, shutil
import json
import cv2
from . import batch

LONG_SIDE = 1024
JPEG_QUALITY = 95
CLASS_NAMES = ("leak",)
LABEL_FOLDERS = ("labels", "rejected", "no_leak")
LABEL_SUFFIXES = (".label.json", ".boxes.json", ".json")
SPLITS = ("train", "val", "test")
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
MAX_TRAIN_ONLY_GROUP = 3
SPLIT_COLUMNS = ("stem", "split", "orientation", "status")
FROZEN_SPLIT_NAME = "split.csv"
OUTPUT_FOLDERS = ("images", "labels")


def label_stem(path):
    name = os.path.basename(path)
    return name[:-len(next(s for s in LABEL_SUFFIXES if name.endswith(s)))]


def frame_size(label):
    width, height = label["size"].lower().split("x")
    return int(width), int(height)


def raw_leak_boxes(label):
    if "leak_boxes" in label: return [tuple(int(v) for v in b) for b in label["leak_boxes"]]
    return [tuple(int(v) for v in b) for st in label.get("stages", []) if st["stage"] in ("recipe_v2", "leak") for b in st["boxes"]]


def label_status(label, folder, boxes):
    if not boxes: return "no-leak"
    return label.get("status") or {"labels": "accepted", "rejected": "rejected", "no_leak": "no-leak"}[folder]


def read_scan_record(path, folder):
    label = json.load(open(path)); width, height = frame_size(label); boxes = raw_leak_boxes(label)
    return {"stem": label_stem(path), "image": label["image"], "width": width, "height": height,
            "boxes": boxes, "status": label_status(label, folder, boxes)}


def dataset_records(dataset_dir):
    records = []
    for folder in LABEL_FOLDERS:
        for path in sorted(glob.glob(os.path.join(dataset_dir, folder, "*.json"))): records.append(read_scan_record(path, folder))
    stems = [r["stem"] for r in records]
    duplicated = sorted({s for s in stems if stems.count(s) > 1})
    if duplicated: raise ValueError(f"scan(s) labelled in more than one folder of {dataset_dir}: {', '.join(duplicated)}")
    return records


def box_area(box):
    x0, x1, y0, y1 = box
    return (x1 - x0) * (y1 - y0)


def orientation(boxes, width, height):
    if not boxes: return "none"
    x0, x1, y0, y1 = max((clip_box(b, width, height) for b in boxes), key=box_area)
    if y1 - y0 > x1 - x0: return "edge" if x0 == 0 or x1 == width else "vertical"
    return "edge" if y0 == 0 or y1 == height else "horizontal"


def split_sizes(count, fractions=SPLIT_FRACTIONS):
    if count <= MAX_TRAIN_ONLY_GROUP: return count, 0, 0
    val, test = (max(1, round(count * f)) for f in fractions[1:])
    return count - val - test, val, test


def read_frozen_rows(path):
    if not os.path.isfile(path): return []
    return list(csv.DictReader(open(path, newline="")))


def read_frozen_split(path):
    frozen = {row["stem"]: row["split"] for row in read_frozen_rows(path)}
    unknown = sorted({split for split in frozen.values() if split not in SPLITS})
    if unknown: raise ValueError(f"{path}: unknown split name(s) {', '.join(unknown)}; expected {', '.join(SPLITS)}")
    return frozen


def assign_splits(records, seed=0, fractions=SPLIT_FRACTIONS, frozen=None):
    frozen = frozen or {}
    split_of = {r["stem"]: frozen[r["stem"]] for r in records if r["stem"] in frozen}
    split_of.update(assign_new_splits([r for r in records if r["stem"] not in frozen], seed, fractions))
    return split_of


def assign_new_splits(records, seed, fractions):
    groups = {}
    for r in records: groups.setdefault((orientation(r["boxes"], r["width"], r["height"]), r["status"]), []).append(r["stem"])
    split_of = {}
    for (orient, status), stems in sorted(groups.items()):
        shuffled = sorted(stems); random.Random(f"{seed}:{orient}:{status}").shuffle(shuffled)
        train, val, _ = split_sizes(len(shuffled), fractions)
        for i, stem in enumerate(shuffled): split_of[stem] = "train" if i < train else "val" if i < train + val else "test"
    return split_of


def clip_box(box, width, height):
    x0, x1, y0, y1 = box
    return (min(max(0, x0), width), min(max(0, x1), width), min(max(0, y0), height), min(max(0, y1), height))


def yolo_line(box, width, height, class_id=0):
    x0, x1, y0, y1 = clip_box(box, width, height)
    return f"{class_id} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} {(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}"


def yolo_text(boxes, width, height):
    kept = [b for b in (clip_box(b, width, height) for b in boxes) if b[1] > b[0] and b[3] > b[2]]
    return "".join(yolo_line(b, width, height) + "\n" for b in kept)


def resized_size(width, height, long_side=LONG_SIDE):
    scale = long_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def resize_for_yolo(bgr, long_side=LONG_SIDE):
    height, width = bgr.shape[:2]
    return cv2.resize(bgr, resized_size(width, height, long_side), interpolation=cv2.INTER_AREA)


def read_original(original_dir, record):
    path = os.path.join(original_dir, record["image"])
    if not os.path.isfile(path): path = batch.find_scan(original_dir, record["stem"])
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None: raise ValueError(f"not a readable image: {path}")
    if (bgr.shape[1], bgr.shape[0]) != (record["width"], record["height"]):
        raise ValueError(f"{record['stem']}: label size {record['width']}x{record['height']} does not match the image {bgr.shape[1]}x{bgr.shape[0]}")
    return bgr


def data_yaml_text(out_dir):
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    return f"path: {os.path.abspath(out_dir)}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n"


def write_scan(out_dir, split, record, bgr):
    image_dir, label_dir = os.path.join(out_dir, "images", split), os.path.join(out_dir, "labels", split)
    os.makedirs(image_dir, exist_ok=True); os.makedirs(label_dir, exist_ok=True)
    cv2.imwrite(os.path.join(image_dir, record["stem"] + ".jpg"), resize_for_yolo(bgr), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    with open(os.path.join(label_dir, record["stem"] + ".txt"), "w") as f: f.write(yolo_text(record["boxes"], record["width"], record["height"]))


def split_rows(records, split_of):
    return [{"stem": r["stem"], "split": split_of[r["stem"]], "orientation": orientation(r["boxes"], r["width"], r["height"]), "status": r["status"]} for r in records]


def clear_outputs(out_dir):
    for folder in OUTPUT_FOLDERS:
        path = os.path.join(out_dir, folder)
        if os.path.isdir(path): shutil.rmtree(path)


def write_split_csv(path, rows):
    with open(path + ".tmp", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SPLIT_COLUMNS); writer.writeheader(); writer.writerows(rows)
    os.replace(path + ".tmp", path)


def build_yolo_dataset(dataset_dir, out_dir, original_dir=batch.DEFAULT_ORIGINAL, seed=0, split_path=None):
    split_path = split_path or os.path.join(dataset_dir, FROZEN_SPLIT_NAME)
    records = dataset_records(dataset_dir); split_of = assign_splits(records, seed, frozen=read_frozen_split(split_path))
    clear_outputs(out_dir)
    for split in SPLITS:
        for kind in ("images", "labels"): os.makedirs(os.path.join(out_dir, kind, split), exist_ok=True)
    for r in records: write_scan(out_dir, split_of[r["stem"]], r, read_original(original_dir, r))
    with open(os.path.join(out_dir, "data.yaml"), "w") as f: f.write(data_yaml_text(out_dir))
    rows = split_rows(records, split_of)
    write_split_csv(os.path.join(out_dir, "split.csv"), rows); write_split_csv(split_path, rows_with_absent_frozen(rows, read_frozen_rows(split_path)))
    return rows


def rows_with_absent_frozen(rows, frozen_rows):
    present = {r["stem"] for r in rows}
    return rows + [{k: r.get(k, "") for k in SPLIT_COLUMNS} for r in frozen_rows if r["stem"] not in present]


def split_summary(rows):
    counts = {s: sum(r["split"] == s for r in rows) for s in SPLITS}
    return f"{len(rows)} scans: " + ", ".join(f"{s} {n}" for s, n in counts.items())


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m leakfix.yolo_data")
    p.add_argument("--dataset", default="dataset"); p.add_argument("--out", default="work/yolo")
    p.add_argument("--original", default=batch.DEFAULT_ORIGINAL); p.add_argument("--seed", type=int, default=0, help="seed for stems not yet in the frozen split")
    p.add_argument("--split", default=None, help="frozen split file, default <dataset>/split.csv")
    a = p.parse_args(argv)
    try:
        rows = build_yolo_dataset(a.dataset, a.out, a.original, a.seed, a.split)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(str(e))
    print(split_summary(rows)); print(f"wrote {a.out}/data.yaml, {a.out}/split.csv and the frozen split {a.split or os.path.join(a.dataset, FROZEN_SPLIT_NAME)}")


if __name__ == "__main__":
    main()
