import argparse, csv, glob, importlib.util, os
import numpy as np, cv2
from . import core

PRIMARY_MODEL = "yolo26m.pt"
FALLBACK_MODEL = "yolo11m.pt"
DEFAULT_WEIGHTS = "work/detect/weights/best.pt"
IMGSZ = 1024
EPOCHS = 150
PATIENCE = 30
DEFAULT_CONF = 0.25
CLASS_NAME = "leak"
INSTALL_HINT = "install .[detect] (pip install -e '.[detect]')"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
GEOMETRIC_ONLY_SETTINGS = {
    "fliplr": 0.5, "flipud": 0.5, "degrees": 0.0, "translate": 0.0, "scale": 0.0, "shear": 0.0, "perspective": 0.0,
    "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
}
COLOUR_SETTING_KEYS = ("hsv_h", "hsv_s", "hsv_v")


def available():
    return importlib.util.find_spec("ultralytics") is not None


def colour_augmenting_packages():
    return [name for name in ("albumentations",) if importlib.util.find_spec(name) is not None]


def require_ultralytics():
    if not available(): raise RuntimeError(f"ultralytics is not installed: {INSTALL_HINT}")
    from ultralytics import YOLO
    return YOLO


def load_detector(weights):
    YOLO = require_ultralytics()
    if not os.path.isfile(weights): raise FileNotFoundError(f"detector weights not found: {weights}")
    return YOLO(weights)


def load_base_model(model, fallback=FALLBACK_MODEL):
    YOLO = require_ultralytics()
    try:
        return YOLO(model), model
    except Exception as e:
        if model == fallback: raise
        print(f"{model} unavailable ({e}); falling back to {fallback}")
        return YOLO(fallback), fallback


def read_data_yaml(path):
    fields = {}
    for line in open(path):
        if ":" in line and not line.startswith(" "):
            key, value = line.split(":", 1); fields[key.strip()] = value.strip()
    return fields


def split_image_dir(data_yaml, split):
    fields = read_data_yaml(data_yaml)
    return os.path.join(fields.get("path") or os.path.dirname(os.path.abspath(data_yaml)), fields[split])


def label_path_for_image(image_path):
    split_folder, name = os.path.split(image_path)
    root = os.path.dirname(os.path.dirname(split_folder))
    return os.path.join(root, "labels", os.path.basename(split_folder), os.path.splitext(name)[0] + ".txt")


def split_images(image_dir):
    return [p for p in sorted(glob.glob(os.path.join(image_dir, "*"))) if p.lower().endswith(IMAGE_EXTENSIONS)]


def rotate_yolo_line_clockwise(line):
    class_id, cx, cy, w, h = line.split()
    return f"{class_id} {1 - float(cy):.6f} {float(cx):.6f} {float(h):.6f} {float(w):.6f}"


def rotate_yolo_text_clockwise(text):
    return "".join(rotate_yolo_line_clockwise(line) + "\n" for line in text.splitlines() if line.strip())


def write_rotated_train_split(data_yaml, out_dir):
    train_dir = split_image_dir(data_yaml, "train")
    image_out, label_out = os.path.join(out_dir, "rot90", "images", "train"), os.path.join(out_dir, "rot90", "labels", "train")
    os.makedirs(image_out, exist_ok=True); os.makedirs(label_out, exist_ok=True)
    for image_path in split_images(train_dir):
        stem = os.path.splitext(os.path.basename(image_path))[0] + "_r90"
        cv2.imwrite(os.path.join(image_out, stem + ".png"), cv2.rotate(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.ROTATE_90_CLOCKWISE))
        label_path = label_path_for_image(image_path)
        text = open(label_path).read() if os.path.isfile(label_path) else ""
        with open(os.path.join(label_out, stem + ".txt"), "w") as f: f.write(rotate_yolo_text_clockwise(text))
    return os.path.abspath(train_dir), os.path.abspath(image_out)


def rotated_data_yaml_text(data_yaml, train_dirs):
    val, test = (os.path.abspath(split_image_dir(data_yaml, s)) for s in ("val", "test"))
    return f"train: [{', '.join(train_dirs)}]\nval: {val}\ntest: {test}\nnames:\n  0: {CLASS_NAME}\n"


def training_settings():
    return dict(GEOMETRIC_ONLY_SETTINGS)


def train(data_yaml, out, model=PRIMARY_MODEL, imgsz=IMGSZ, epochs=EPOCHS, patience=PATIENCE):
    blocked = colour_augmenting_packages()
    if blocked: raise RuntimeError(f"{', '.join(blocked)} is installed and ultralytics would apply its colour transforms; uninstall it before training (invariant 4)")
    detector, used_model = load_base_model(model)
    out = os.path.abspath(out); os.makedirs(out, exist_ok=True)
    train_dirs = write_rotated_train_split(data_yaml, out)
    rotated_yaml = os.path.join(out, "data_rot90.yaml")
    with open(rotated_yaml, "w") as f: f.write(rotated_data_yaml_text(data_yaml, train_dirs))
    detector.train(data=rotated_yaml, imgsz=imgsz, epochs=epochs, patience=patience, project=os.path.dirname(out), name=os.path.basename(out), exist_ok=True, **training_settings())
    return os.path.join(out, "weights", "best.pt"), used_model


def yolo_boxes_from_text(text, width, height):
    boxes = []
    for line in text.splitlines():
        if not line.strip(): continue
        _, cx, cy, w, h = (float(v) for v in line.split())
        boxes.append(((cx - w / 2) * width, (cx + w / 2) * width, (cy - h / 2) * height, (cy + h / 2) * height))
    return boxes


def box_iou(a, b):
    ix = max(0.0, min(a[1], b[1]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = ix * iy; union = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / union if union > 0 else 0.0


def best_match_ious(labelled, predicted):
    return [max((box_iou(box, p) for p in predicted), default=0.0) for box in labelled]


def full_res_box(xyxy, shape):
    x0, y0, x1, y1 = xyxy
    return core.clamp_box((int(np.floor(x0)), int(np.ceil(x1)), int(np.floor(y0)), int(np.ceil(y1))), shape)


def boxes_from_result(result, shape):
    found = []
    for xyxy, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
        try: found.append((full_res_box(xyxy, shape), float(confidence)))
        except ValueError: continue
    return found


def bgr_u8(rgb):
    return cv2.cvtColor((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)


def propose_rgb(detector, rgb, conf=DEFAULT_CONF, imgsz=IMGSZ):
    result = detector.predict(bgr_u8(rgb), imgsz=imgsz, conf=conf, verbose=False)[0]
    return boxes_from_result(result, rgb.shape)


def propose(weights, scan_path, conf=DEFAULT_CONF, imgsz=IMGSZ):
    return propose_rgb(load_detector(weights), core.load(scan_path), conf, imgsz)


def evaluate(weights, data_yaml, imgsz=IMGSZ, conf=DEFAULT_CONF):
    detector = load_detector(weights)
    metrics = detector.val(data=data_yaml, split="test", imgsz=imgsz, plots=False)
    per_scan = []
    for image_path in split_images(split_image_dir(data_yaml, "test")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR); height, width = image.shape[:2]
        label_path = label_path_for_image(image_path)
        labelled = yolo_boxes_from_text(open(label_path).read() if os.path.isfile(label_path) else "", width, height)
        result = detector.predict(image, imgsz=imgsz, conf=conf, verbose=False)[0]
        predicted = [tuple(float(v) for v in (x0, x1, y0, y1)) for x0, y0, x1, y1 in result.boxes.xyxy.cpu().numpy()]
        per_scan.append({"scan": os.path.splitext(os.path.basename(image_path))[0], "labelled": len(labelled), "predicted": len(predicted),
                         "best_ious": [round(v, 3) for v in best_match_ious(labelled, predicted)]})
    return {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map), "per_scan": per_scan}


def write_per_scan_csv(path, per_scan):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f); writer.writerow(("scan", "labelled", "predicted", "best_ious"))
        for row in per_scan: writer.writerow((row["scan"], row["labelled"], row["predicted"], " ".join(str(v) for v in row["best_ious"])))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m leakfix.detect"); sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train"); t.add_argument("data_yaml"); t.add_argument("--out", default="work/detect"); t.add_argument("--model", default=PRIMARY_MODEL)
    t.add_argument("--imgsz", type=int, default=IMGSZ); t.add_argument("--epochs", type=int, default=EPOCHS); t.add_argument("--patience", type=int, default=PATIENCE)
    e = sub.add_parser("evaluate"); e.add_argument("weights"); e.add_argument("data_yaml"); e.add_argument("--imgsz", type=int, default=IMGSZ)
    e.add_argument("--conf", type=float, default=DEFAULT_CONF); e.add_argument("--csv", default=None)
    r = sub.add_parser("propose"); r.add_argument("weights"); r.add_argument("scan"); r.add_argument("--conf", type=float, default=DEFAULT_CONF)
    a = p.parse_args(argv)
    try:
        if a.command == "train":
            best, used = train(a.data_yaml, a.out, a.model, a.imgsz, a.epochs, a.patience); print(f"trained {used}; best weights {best}")
        elif a.command == "evaluate":
            report = evaluate(a.weights, a.data_yaml, a.imgsz, a.conf)
            print(f"test mAP50 {report['map50']:.3f}  mAP50-95 {report['map50_95']:.3f}")
            for row in report["per_scan"]: print(f"{row['scan']}: {row['labelled']} labelled, {row['predicted']} predicted, best IoU per labelled box {row['best_ious']}")
            if a.csv: write_per_scan_csv(a.csv, report["per_scan"]); print("saved", a.csv)
        elif a.command == "propose":
            for box, confidence in propose(a.weights, a.scan, a.conf): print(f"{CLASS_NAME} {box} {confidence:.2f}")
    except (RuntimeError, FileNotFoundError, ValueError) as err:
        raise SystemExit(str(err))


if __name__ == "__main__":
    main()
