import argparse, base64, glob, importlib.util, json, os, sys, time
from types import SimpleNamespace
import numpy as np, cv2
from . import compare

DEFAULT_PROMPT = "Remove the light leak from this, but keep the film aspect"
KEY_ENV = "GEMINI_API_KEY"
INSTALL_HINT = "pip install -e .[gemini]"
ORIGINAL_DIR = "photos/original"
FIXED_DIR = "dataset/fixed"
DATASET_DIR = "dataset"
LABEL_FOLDERS = ("labels", "rejected")
OUT_DIR = "work/gemini"
COMPARE_CSV_NAME = "compare.csv"
RAW_LABEL_PREFIX = "gemini"
PRE_LABEL_PREFIX = "pre+gemini"
RATE_LIMIT_CODE = 429
RETRY_BACKOFF_S = 30.0
RESPONSE_MODALITIES = ("TEXT", "IMAGE")
IMAGE_SIZES = ("1K", "2K", "4K")
DEFAULT_IMAGE_SIZE = "2K"
IMAGE_ACTION = "generateContent"
MODEL_PREFIX = "models/"
GEMINI_ID_PREFIX = "gemini-"
MIME_BY_EXTENSION = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
BATCH_JOB_NAME = "batch_job.json"
BATCH_REQUESTS_DIR = "batch"
BATCH_UPLOAD_MIME = "jsonl"
BATCH_DISPLAY_PREFIX = "leakfix-gemini"
BATCH_POLL_S = 60.0
BATCH_DONE_STATES = ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED")
BATCH_FAILED_STATES = ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED")
COLLECT_COMMAND = "collect"
UNREADABLE_OUTCOME = "unreadable"


def available():
    try:
        return importlib.util.find_spec("google.genai") is not None
    except ModuleNotFoundError:
        return False


def require_genai():
    if not available(): raise RuntimeError(f"google-genai is not installed: {INSTALL_HINT}")


def api_key_from_env():
    key = os.environ.get(KEY_ENV, "").strip()
    if not key: raise RuntimeError(f"no API key: set {KEY_ENV} in the environment")
    return key


def make_client():
    require_genai()
    from google import genai
    return genai.Client(api_key=api_key_from_env())


def bare_model_id(model_id):
    return model_id[len(MODEL_PREFIX):] if model_id.startswith(MODEL_PREFIX) else model_id


def short_model_name(model_id):
    safe = compare.filename_safe(bare_model_id(model_id))
    return safe[len(GEMINI_ID_PREFIX):] if safe.startswith(GEMINI_ID_PREFIX) and len(safe) > len(GEMINI_ID_PREFIX) else safe


def raw_label(model_id): return f"{RAW_LABEL_PREFIX}-{short_model_name(model_id)}"


def pre_label(model_id): return f"{PRE_LABEL_PREFIX}-{short_model_name(model_id)}"


def catalogue_entry(model):
    return {"id": bare_model_id(model.name or ""), "actions": list(model.supported_actions or [])}


def model_catalogue(client):
    return [catalogue_entry(m) for m in client.models.list()]


def image_capable_ids(catalogue):
    return sorted(e["id"] for e in catalogue if "image" in e["id"].lower() or IMAGE_ACTION in e["actions"])


def image_capable_text(catalogue):
    ids = image_capable_ids(catalogue)
    return "image-capable ids: " + (", ".join(ids) if ids else "(none listed for this key)")


def checked_models(requested, catalogue):
    if not requested: raise ValueError("give at least one --model <id> (" + image_capable_text(catalogue) + ")")
    known = {e["id"] for e in catalogue}
    unknown = [m for m in requested if bare_model_id(m) not in known]
    if unknown: raise ValueError(f"unknown model id(s): {', '.join(unknown)}; " + image_capable_text(catalogue))
    return list(dict.fromkeys(bare_model_id(m) for m in requested))


def read_stems(stems, stems_file):
    names = [s.strip() for group in stems or [] for s in group.split(",") if s.strip()]
    if stems_file: names += [line.strip() for line in open(stems_file, encoding="utf-8") if line.strip() and not line.startswith("#")]
    if not names: raise ValueError("give --stems a,b,... or --stems-file")
    return list(dict.fromkeys(names))


def stem_jobs(stem, model_id, original_dir=ORIGINAL_DIR, fixed_dir=FIXED_DIR):
    return [(raw_label(model_id), os.path.join(original_dir, f"{stem}.jpg")),
            (pre_label(model_id), os.path.join(fixed_dir, f"{stem}_fixed.jpg"))]


def output_paths(out_dir, stem, label):
    folder = os.path.join(out_dir, stem)
    return os.path.join(folder, f"{label}.png"), os.path.join(folder, f"{label}.json")


def pending_jobs(stems, model_ids, out_dir, original_dir, fixed_dir, force, report=print):
    pending, outcomes = [], {}
    for stem in stems:
        for model_id in model_ids:
            for label, input_path in stem_jobs(stem, model_id, original_dir, fixed_dir):
                if os.path.exists(output_paths(out_dir, stem, label)[0]) and not force:
                    report(f"{stem} {label}: exists, skipped (--force to redo)"); outcomes["exists"] = outcomes.get("exists", 0) + 1
                elif not os.path.isfile(input_path):
                    report(f"{stem} {label}: input missing, skipped: {input_path}"); outcomes["missing-input"] = outcomes.get("missing-input", 0) + 1
                else:
                    pending.append({"key": f"{stem}/{label}", "stem": stem, "label": label, "model": model_id, "input": input_path})
    return pending, outcomes


def mime_type_for(path):
    return MIME_BY_EXTENSION.get(os.path.splitext(path)[1].lower(), "image/jpeg")


def request_contents(input_path, prompt):
    from google.genai import types
    return [types.Part.from_bytes(data=open(input_path, "rb").read(), mime_type=mime_type_for(input_path)), prompt]


def request_config(image_size=DEFAULT_IMAGE_SIZE):
    from google.genai import types
    return types.GenerateContentConfig(response_modalities=list(RESPONSE_MODALITIES), image_config=types.ImageConfig(image_size=image_size))


def batch_request_line(key, input_path, prompt, image_size=DEFAULT_IMAGE_SIZE):
    image_part = {"inlineData": {"mimeType": mime_type_for(input_path), "data": base64.b64encode(open(input_path, "rb").read()).decode("ascii")}}
    return {"key": key, "request": {"contents": [{"role": "user", "parts": [image_part, {"text": prompt}]}],
                                    "generationConfig": {"responseModalities": list(RESPONSE_MODALITIES), "imageConfig": {"imageSize": image_size}}}}


def response_parts(response):
    candidates = getattr(response, "candidates", None) or []
    content = getattr(candidates[0], "content", None) if candidates else None
    return list(getattr(content, "parts", None) or [])


def response_image_bytes(response):
    for part in response_parts(response):
        blob = getattr(part, "inline_data", None)
        if blob is not None and blob.data and str(blob.mime_type or "").startswith("image/"): return blob.data
    return None


def response_text(response):
    return "\n".join(part.text for part in response_parts(response) if getattr(part, "text", None) and not getattr(part, "thought", False))


def enum_text(value):
    return str(getattr(value, "name", value)) if value is not None else None


def finish_reason(response):
    candidates = getattr(response, "candidates", None) or []
    return enum_text(getattr(candidates[0], "finish_reason", None) if candidates else None)


def block_reason(response):
    return enum_text(getattr(getattr(response, "prompt_feedback", None), "block_reason", None))


def is_rate_limit(error):
    return getattr(error, "code", None) == RATE_LIMIT_CODE


def generate_with_retry(client, model_id, contents, config, report=print, sleep=time.sleep):
    from google.genai import errors
    try:
        return client.models.generate_content(model=model_id, contents=contents, config=config)
    except errors.APIError as e:
        if not is_rate_limit(e): raise
        report(f"  rate limited by {model_id}; retrying once in {RETRY_BACKOFF_S:.0f} s")
        sleep(RETRY_BACKOFF_S)
        return client.models.generate_content(model=model_id, contents=contents, config=config)


def png_bytes(image_bytes):
    decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None: return None
    ok, encoded = cv2.imencode(".png", decoded)
    return encoded.tobytes() if ok else None


def write_bytes_atomic(path, data):
    temp_path = path + ".tmp"
    with open(temp_path, "wb") as f: f.write(data)
    os.replace(temp_path, path)


def run_record(job, prompt, image_size, text, status, reason=None, blocked=None, batch_job=None):
    record = {"model": job["model"], "label": job["label"], "prompt": prompt, "input": job["input"], "size": image_size,
              "time": time.strftime("%Y-%m-%d %H:%M:%S"), "status": status, "finish_reason": reason, "block_reason": blocked, "text": text}
    if batch_job: record["batch_job"] = batch_job
    return record


def write_record(path, record):
    write_bytes_atomic(path, json.dumps(record, indent=2).encode("utf-8"))


def save_response(response, job, prompt, image_size, out_dir, catalogue, report=print, batch_job=None):
    image_path, record_path = output_paths(out_dir, job["stem"], job["label"])
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image, text, reason, blocked = response_image_bytes(response), response_text(response), finish_reason(response), block_reason(response)
    png = png_bytes(image) if image else None
    if png is None:
        write_record(record_path, run_record(job, prompt, image_size, text, "no-image", reason, blocked, batch_job))
        report(f"{job['stem']} {job['label']}: no image in the response (finish reason {reason}, block reason {blocked}), skipped; {image_capable_text(catalogue)}"); return "no-image"
    write_bytes_atomic(image_path, png)
    write_record(record_path, run_record(job, prompt, image_size, text, "ok", reason, blocked, batch_job))
    report(f"{job['stem']} {job['label']}: wrote {image_path}"); return "ok"


def run_job(client, job, prompt, image_size, out_dir, catalogue, report=print, sleep=time.sleep):
    response = generate_with_retry(client, job["model"], request_contents(job["input"], prompt), request_config(image_size), report, sleep)
    return save_response(response, job, prompt, image_size, out_dir, catalogue, report)


def count(outcomes, outcome):
    outcomes[outcome] = outcomes.get(outcome, 0) + 1


def generate(client, stems, model_ids, prompt=DEFAULT_PROMPT, out_dir=OUT_DIR, original_dir=ORIGINAL_DIR, fixed_dir=FIXED_DIR,
             catalogue=(), force=False, image_size=DEFAULT_IMAGE_SIZE, report=print, sleep=time.sleep):
    from google.genai import errors
    pending, outcomes = pending_jobs(stems, model_ids, out_dir, original_dir, fixed_dir, force, report)
    for job in pending:
        try:
            count(outcomes, run_job(client, job, prompt, image_size, out_dir, catalogue, report, sleep))
        except errors.APIError as e:
            report(f"{job['stem']} {job['label']}: API error {getattr(e, 'code', '?')}: {getattr(e, 'message', None) or e}, skipped"); count(outcomes, "api-error")
    return outcomes


def batch_job_path(out_dir):
    return os.path.join(out_dir, BATCH_JOB_NAME)


def read_batch_record(out_dir):
    path = batch_job_path(out_dir)
    if not os.path.isfile(path): raise ValueError(f"no batch job recorded at {path}; submit one with --batch first")
    return json.load(open(path))


def uncollected_jobs(record):
    return [job for job in record.get("jobs", []) if not job.get("collected")]


def refuse_overwriting_open_batch(out_dir, force):
    path = batch_job_path(out_dir)
    if force or not os.path.isfile(path): return
    open_jobs = uncollected_jobs(json.load(open(path)))
    if open_jobs: raise ValueError(f"{path} still has {len(open_jobs)} uncollected job(s) ({', '.join(j['name'] for j in open_jobs)}); run `collect` first, or --force to replace the record")


def write_requests_file(path, jobs, prompt, image_size):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        for job in jobs: f.write(json.dumps(batch_request_line(job["key"], job["input"], prompt, image_size)) + "\n")
    os.replace(path + ".tmp", path)
    return path


def jobs_by_model(pending):
    grouped = {}
    for job in pending: grouped.setdefault(job["model"], []).append(job)
    return grouped


def submit_batch(client, stems, model_ids, prompt=DEFAULT_PROMPT, out_dir=OUT_DIR, original_dir=ORIGINAL_DIR, fixed_dir=FIXED_DIR,
                 force=False, image_size=DEFAULT_IMAGE_SIZE, report=print):
    from google.genai import types
    refuse_overwriting_open_batch(out_dir, force)
    pending, outcomes = pending_jobs(stems, model_ids, out_dir, original_dir, fixed_dir, force, report)
    record = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "prompt": prompt, "size": image_size, "stems": list(stems), "jobs": []}
    for model_id, jobs in jobs_by_model(pending).items():
        requests_path = write_requests_file(os.path.join(out_dir, BATCH_REQUESTS_DIR, f"{short_model_name(model_id)}.jsonl"), jobs, prompt, image_size)
        display_name = f"{BATCH_DISPLAY_PREFIX}-{short_model_name(model_id)}"
        uploaded = client.files.upload(file=requests_path, config=types.UploadFileConfig(display_name=display_name, mime_type=BATCH_UPLOAD_MIME))
        batch = client.batches.create(model=model_id, src=uploaded.name, config=types.CreateBatchJobConfig(display_name=display_name))
        record["jobs"].append({"name": batch.name, "model": model_id, "requests_file": uploaded.name, "requests_path": requests_path,
                               "state": enum_text(batch.state), "collected": False, "entries": jobs})
        report(f"submitted {batch.name} ({model_id}, {len(jobs)} request(s))")
        outcomes["submitted"] = outcomes.get("submitted", 0) + len(jobs)
    if record["jobs"]: write_bytes_atomic(batch_job_path(out_dir), json.dumps(record, indent=2).encode("utf-8"))
    return record, outcomes


def batch_result_lines(client, batch):
    dest = getattr(batch, "dest", None)
    if dest is not None and dest.file_name:
        return [json.loads(line) for line in client.files.download(file=dest.file_name).decode("utf-8").splitlines() if line.strip()]
    return [{"key": (r.metadata or {}).get("key"), "response_object": r.response, "error": r.error} for r in (getattr(dest, "inlined_responses", None) or [])]


def rest_field(mapping, camel, snake):
    if not isinstance(mapping, dict): return None
    return mapping.get(camel, mapping.get(snake))


def blob_bytes(data):
    if isinstance(data, (bytes, bytearray)): return bytes(data)
    return base64.b64decode(data) if data else None


def part_from_rest(part):
    blob = rest_field(part, "inlineData", "inline_data")
    inline = SimpleNamespace(data=blob_bytes(rest_field(blob, "data", "data")), mime_type=rest_field(blob, "mimeType", "mime_type")) if blob else None
    return SimpleNamespace(text=part.get("text"), thought=part.get("thought", False), inline_data=inline)


def candidate_from_rest(candidate):
    content = candidate.get("content") if isinstance(candidate, dict) else None
    parts = [part_from_rest(p) for p in (content.get("parts") or []) if isinstance(p, dict)] if isinstance(content, dict) else []
    return SimpleNamespace(content=SimpleNamespace(parts=parts) if content is not None else None, finish_reason=rest_field(candidate, "finishReason", "finish_reason"))


def response_from_rest(response):
    feedback = rest_field(response, "promptFeedback", "prompt_feedback")
    return SimpleNamespace(candidates=[candidate_from_rest(c) for c in (response.get("candidates") or [])],
                           prompt_feedback=SimpleNamespace(block_reason=rest_field(feedback, "blockReason", "block_reason")) if feedback else None)


def response_from_line(line):
    if line.get("response_object") is not None: return line["response_object"]
    return response_from_rest(line["response"]) if isinstance(line.get("response"), dict) else None


def save_batch_results(client, batch, job_record, prompt, image_size, out_dir, catalogue, report=print):
    entries = {entry["key"]: entry for entry in job_record["entries"]}
    outcomes = {}
    for line in batch_result_lines(client, batch):
        job = entries.pop(line.get("key"), None)
        if job is None: report(f"{job_record['name']}: result with unknown key {line.get('key')!r} ignored"); continue
        try:
            response = response_from_line(line)
            if response is None:
                report(f"{job['stem']} {job['label']}: batch error {line.get('error')}, skipped"); count(outcomes, "api-error"); continue
            count(outcomes, save_response(response, job, prompt, image_size, out_dir, catalogue, report, batch_job=job_record["name"]))
        except Exception as e:
            report(f"{job['stem']} {job['label']}: could not read this result ({type(e).__name__}: {e}), skipped"); count(outcomes, UNREADABLE_OUTCOME)
    for job in entries.values():
        report(f"{job['stem']} {job['label']}: no result in {job_record['name']}, skipped"); count(outcomes, "no-result")
    return outcomes


def merge_outcomes(total, more):
    for name, n in (more or {}).items(): total[name] = total.get(name, 0) + n
    return total


def collected_outcomes(client, batch, job_record, record, out_dir, catalogue, report=print):
    try:
        return save_batch_results(client, batch, job_record, record["prompt"], record["size"], out_dir, catalogue, report)
    except Exception as e:
        report(f"{job_record['name']} ({job_record['model']}): could not download or read the results ({type(e).__name__}: {e})"); return None


def collect_batch(client, out_dir=OUT_DIR, wait=False, catalogue=(), report=print, sleep=time.sleep, poll_s=BATCH_POLL_S):
    record = read_batch_record(out_dir); outcomes = {}
    while True:
        still_running = []
        for job_record in uncollected_jobs(record):
            batch = client.batches.get(name=job_record["name"]); state = enum_text(batch.state); job_record["state"] = state
            if state in BATCH_DONE_STATES:
                job_outcomes = collected_outcomes(client, batch, job_record, record, out_dir, catalogue, report)
                merge_outcomes(outcomes, job_outcomes)
                job_record["collected"] = job_outcomes is not None and not job_outcomes.get(UNREADABLE_OUTCOME)
                if not job_record["collected"]: report(f"{job_record['name']} ({job_record['model']}): left uncollected so the next `collect` downloads it again")
            elif state in BATCH_FAILED_STATES:
                report(f"{job_record['name']} ({job_record['model']}): {state}, nothing to collect"); job_record["collected"] = True
                outcomes["batch-failed"] = outcomes.get("batch-failed", 0) + len(job_record["entries"])
            else:
                still_running.append(job_record); report(f"{job_record['name']} ({job_record['model']}): {state}")
        write_bytes_atomic(batch_job_path(out_dir), json.dumps(record, indent=2).encode("utf-8"))
        if not still_running or not wait: return record, outcomes, still_running
        report(f"{len(still_running)} job(s) still running; checking again in {poll_s:.0f} s"); sleep(poll_s)


def label_path_for(stem, dataset_dir=DATASET_DIR):
    for folder in LABEL_FOLDERS:
        path = os.path.join(dataset_dir, folder, f"{stem}.boxes.json")
        if os.path.isfile(path): return path
    return None


def stem_versions(out_dir, stem):
    return [(os.path.splitext(os.path.basename(p))[0], p) for p in sorted(glob.glob(os.path.join(out_dir, stem, "*.png")))]


def compare_stem(stem, out_dir=OUT_DIR, original_dir=ORIGINAL_DIR, dataset_dir=DATASET_DIR, report=print):
    versions = stem_versions(out_dir, stem)
    if not versions: report(f"{stem}: no versions in {os.path.join(out_dir, stem)}, compare skipped"); return None
    label_path = label_path_for(stem, dataset_dir)
    if label_path is None: report(f"{stem}: no boxes.json in {dataset_dir}/labels or rejected, compare skipped"); return None
    try:
        return compare.compare_versions(os.path.join(original_dir, f"{stem}.jpg"), versions, compare.read_label_boxes(label_path),
                                        csv_path=os.path.join(out_dir, COMPARE_CSV_NAME), out_dir=out_dir, report=report)
    except (ValueError, OSError) as e:
        report(f"{stem}: compare failed: {e}"); return None


def outcome_summary(outcomes):
    return ", ".join(f"{n} {name}" for name, n in sorted(outcomes.items())) or "nothing to do"


def add_folder_flags(p):
    p.add_argument("--original-dir", default=ORIGINAL_DIR)
    p.add_argument("--dataset", default=DATASET_DIR, help="dataset folder holding labels/ and rejected/ boxes.json (for compare)")
    p.add_argument("--out", default=OUT_DIR)


def generate_parser():
    p = argparse.ArgumentParser(prog="leakfix.gemini", description="Send raw scans and recipe v2 outputs to Gemini image-editing models and save what comes back. "
                                                                   "`python -m leakfix.gemini collect` downloads a --batch job.")
    p.add_argument("--model", action="append", default=[], help="Gemini model id (repeatable; checked against the models this key can list)")
    p.add_argument("--stems", action="append", default=[], help="comma-separated scan stems (repeatable)")
    p.add_argument("--stems-file", default=None, help="file with one stem per line")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="instruction sent with every image")
    p.add_argument("--size", choices=IMAGE_SIZES, default=DEFAULT_IMAGE_SIZE, help="output image size (image_config.image_size)")
    p.add_argument("--batch", action="store_true", help="submit everything as Gemini Batch API jobs (one per model, half price) instead of calling now")
    p.add_argument("--force", action="store_true", help="regenerate outputs that already exist; with --batch also replace an uncollected batch record")
    p.add_argument("--compare", action="store_true", help="after generating, run leakfix.compare per stem into the output folder")
    p.add_argument("--fixed-dir", default=FIXED_DIR)
    add_folder_flags(p)
    return p


def collect_parser():
    p = argparse.ArgumentParser(prog="leakfix.gemini collect", description="Check the batch jobs in <out>/batch_job.json, save finished results as <out>/<stem>/<label>.png, then run compare.")
    p.add_argument("--wait", action="store_true", help=f"poll every {BATCH_POLL_S:.0f} s until every job has finished")
    p.add_argument("--no-compare", action="store_true", help="skip leakfix.compare after collecting")
    add_folder_flags(p)
    return p


def run_generate(a):
    stems = read_stems(a.stems, a.stems_file)
    client = make_client(); catalogue = model_catalogue(client); model_ids = checked_models(a.model, catalogue)
    if a.batch:
        record, outcomes = submit_batch(client, stems, model_ids, a.prompt, a.out, a.original_dir, a.fixed_dir, a.force, a.size)
        print(f"submitted: {outcome_summary(outcomes)}" + (f"; job names in {batch_job_path(a.out)}; run `python -m leakfix.gemini {COLLECT_COMMAND}` later" if record["jobs"] else ""))
        return
    outcomes = generate(client, stems, model_ids, a.prompt, a.out, a.original_dir, a.fixed_dir, catalogue, a.force, a.size)
    print(f"done: {outcome_summary(outcomes)}")
    if a.compare:
        for stem in stems: compare_stem(stem, a.out, a.original_dir, a.dataset)


def run_collect(a):
    read_batch_record(a.out)
    client = make_client(); catalogue = model_catalogue(client)
    record, outcomes, still_running = collect_batch(client, a.out, a.wait, catalogue)
    open_jobs = uncollected_jobs(record)
    print(f"collected: {outcome_summary(outcomes)}" + (f"; {len(open_jobs)} job(s) not collected yet, run collect again (compare waits until all are)" if open_jobs else ""))
    if open_jobs or a.no_compare: return
    for stem in record.get("stems", []): compare_stem(stem, a.out, a.original_dir, a.dataset)


def api_error_types():
    if not available(): return ()
    from google.genai import errors
    return (errors.APIError,)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    try:
        if argv[:1] == [COLLECT_COMMAND]: run_collect(collect_parser().parse_args(argv[1:]))
        else: run_generate(generate_parser().parse_args(argv))
    except (RuntimeError, ValueError, OSError) + api_error_types() as e:
        sys.exit(f"leakfix.gemini: {e}")


if __name__ == "__main__":
    main()
