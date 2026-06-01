#!/usr/bin/env python3
"""
SmoothQuant accuracy benchmark
================================
Evaluates encoder-style transformer models on SQuAD 1.1 (F1/EM) and
MultiNLI (matched accuracy) under multiple quantization modes:

  none           – float32 or bfloat16 (with --autocast)
  smooth-static  – SmoothQuant, static (activation per-tensor, weight per-row)
  smooth-dynamic – SmoothQuant, dynamic (activation per-row, weight per-row)

Supports torch.compile and AOT Inductor (--aoti).

Usage examples
--------------
# SQuAD accuracy, all modes, alpha sweep
python benchmark_accuracy.py --model bert-large-uncased-whole-word-masking-finetuned-squad \\
    --task squad --quant-mode all --alpha 0.25 0.5 0.75 --autocast

# MultiNLI accuracy with AOTI
python benchmark_accuracy.py --model typeform/distilbert-base-uncased-mnli \\
    --task mnli --quant-mode smooth-dynamic --autocast --aoti

# All accuracy tasks
python benchmark_accuracy.py --model ... --task all --quant-mode all --autocast
"""

import argparse
import json
import os
import re
import string
import sys
import time
import warnings

import torch

warnings.filterwarnings("ignore")

from smoothquant_utils import (
    apply_smoothquant,
    optional_autocast,
    do_compile,
    do_aoti_compile,
    infer as _positional_infer,
    inductor_config,
    _skip_small_linears,
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
ALL_QUANT_MODES = ["fp32", "bf16", "smooth-static", "smooth-static-autocast",
                   "smooth-dynamic", "smooth-dynamic-autocast"]


def parse_args():
    p = argparse.ArgumentParser(
        description="SmoothQuant accuracy benchmark (SQuAD + MultiNLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True,
                   help="HuggingFace model ID")
    p.add_argument("--task", default="squad",
                   choices=["squad", "mnli", "all"],
                   help="Evaluation task  (default: squad)")
    p.add_argument("--quant-mode", type=str, default="fp32",
                   choices=ALL_QUANT_MODES + ["all"],
                   help="Quantization mode  (default: fp32). 'all' runs all four.")
    p.add_argument("--alpha", type=float, nargs="+", default=[0.5], metavar="A",
                   help="SmoothQuant alpha value(s)  (default: 0.5)")
    p.add_argument("--autocast", action="store_true",
                   help="Enable bfloat16 autocast")
    p.add_argument("--aoti", action="store_true",
                   help="Use AOT Inductor instead of torch.compile")
    p.add_argument("--compile", action="store_true",
                   help="Use torch.compile for inference")
    p.add_argument("--num-samples", default="200", metavar="N",
                   help="Validation samples ('all' for entire dataset)  (default: 200)")
    p.add_argument("--num-calib", type=int, default=16,
                   help="Calibration samples for SmoothQuant  (default: 16)")
    p.add_argument("--output", default=None, metavar="FILE",
                   help="Write JSON results to FILE")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_num_samples(raw):
    """Return int sample count, or None for entire dataset."""
    if isinstance(raw, str) and raw.strip().lower() == "all":
        return None
    return int(raw)


def iter_runs(quant_modes, alphas):
    """Yield (quant_mode, alpha, use_autocast, label) for each configuration.

    fp32 and bf16 are baselines (no quantization).
    smooth-static / smooth-dynamic run without autocast.
    smooth-static-autocast / smooth-dynamic-autocast run with bf16 autocast.
    """
    for mode in quant_modes:
        if mode == "fp32":
            yield "none", None, False, "fp32"
        elif mode == "bf16":
            yield "none", None, True, "bf16"
        elif mode.startswith("smooth-"):
            use_autocast = mode.endswith("-autocast")
            base_mode = mode[:-len("-autocast")] if use_autocast else mode
            for a in alphas:
                yield base_mode, a, use_autocast, f"{mode}(a={a:.2f})"
        else:
            yield mode, None, False, mode


def safe_label(s):
    s = re.sub(r"[^A-Za-z0-9_\-]", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def forward(model, inputs, use_autocast, use_aoti=False):
    """Run one forward pass. Handles both dict inputs and AOTI (positional args)."""
    if use_aoti:
        model_inputs = (inputs["input_ids"], inputs["attention_mask"])
        return _positional_infer(model, model_inputs, use_autocast)
    else:
        fn = optional_autocast(lambda m, inp: m(**inp), use_autocast)
        return fn(model, inputs)


def apply_smoothquant_accuracy(model, quant_mode, alpha, calib_inputs,
                               use_autocast=True):
    """Wrapper around smoothquant_utils.apply_smoothquant for accuracy tests.

    Converts dict-based calib_inputs to tuple-based for the utils function.
    Uses _skip_small_linears filter to avoid inductor issues on head layers.
    """
    calib_tuples = [(inp["input_ids"], inp["attention_mask"]) for inp in calib_inputs]
    return apply_smoothquant(model, quant_mode, alpha, calib_tuples,
                             use_autocast=use_autocast,
                             filter_fn=_skip_small_linears)


def maybe_compile_or_aoti(model, args, sample_inputs, use_autocast, label):
    """Compile model with torch.compile or AOTI. Returns (model, is_aoti)."""
    if args.aoti:
        model_inputs = (sample_inputs["input_ids"], sample_inputs["attention_mask"])
        save_path = os.path.join(
            os.getcwd(), f"{safe_label(args.model)}__{safe_label(label)}.pt2")
        compiled = do_aoti_compile(model, model_inputs, use_autocast, save_path)
        print(f"    AOTI package: {save_path}")
        return compiled, True

    elif args.compile:
        compiled = do_compile(model, use_autocast)
        # Trigger compilation
        forward(compiled, sample_inputs, use_autocast, use_aoti=False)
        return compiled, False

    return model, False


# ---------------------------------------------------------------------------
# SQuAD 1.1 evaluation
# ---------------------------------------------------------------------------

def _normalize_answer(s):
    """Lower-case, remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())


def _squad_f1_em(prediction, gold_answers):
    """Return (f1, exact_match) for one prediction vs. gold answers."""
    def _tokens(s): return _normalize_answer(s).split()
    def _f1(pred, gold):
        p_toks, g_toks = _tokens(pred), _tokens(gold)
        common = set(p_toks) & set(g_toks)
        if not common:
            return 0.0
        prec = sum(min(p_toks.count(t), g_toks.count(t)) for t in common) / len(p_toks)
        rec  = sum(min(p_toks.count(t), g_toks.count(t)) for t in common) / len(g_toks)
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    best_f1 = max((_f1(prediction, g) for g in gold_answers), default=0.0)
    best_em = max(
        (1.0 if _normalize_answer(prediction) == _normalize_answer(g) else 0.0
         for g in gold_answers),
        default=0.0,
    )
    return best_f1, best_em


def run_squad(args, quant_modes):
    """Evaluate F1 and Exact Match on SQuAD 1.1 validation set."""
    from datasets import load_dataset
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **_kw): return x

    print(f"\n{'─'*60}")
    print(f" SQuAD 1.1  model={args.model}")
    print(f"{'─'*60}")

    tok = AutoTokenizer.from_pretrained(args.model)
    val_ds = load_dataset("squad", split="validation")
    n = resolve_num_samples(args.num_samples)
    val_ds = val_ds if n is None else val_ds.select(range(min(n, len(val_ds))))
    calib_ds = load_dataset("squad", split="train").select(range(args.num_calib))

    def _encode_qa(question, context):
        return tok(question, context, max_length=384, truncation=True,
                   padding="max_length", return_tensors="pt")

    results = {}
    for quant_mode, alpha, run_autocast, label in iter_runs(quant_modes, args.alpha):
        print(f"\n  [{label}]", flush=True)
        try:
            model = AutoModelForQuestionAnswering.from_pretrained(
                args.model, torch_dtype=torch.float32)
            model.eval()

            if quant_mode != "none":
                calib = []
                for ex in calib_ds:
                    e = _encode_qa(ex["question"], ex["context"])
                    calib.append({"input_ids": e["input_ids"],
                                  "attention_mask": e["attention_mask"]})
                model = apply_smoothquant_accuracy(model, quant_mode, alpha, calib,
                                                   use_autocast=run_autocast)

            # Optionally compile
            sample = _encode_qa("sample question", "sample context")
            sample_dict = {"input_ids": sample["input_ids"],
                           "attention_mask": sample["attention_mask"]}
            model, is_aoti = maybe_compile_or_aoti(
                model, args, sample_dict, run_autocast, f"squad_{label}")

            preds, refs = [], []
            for ex in tqdm(val_ds, desc=label, leave=False):
                enc_out = tok(
                    ex["question"], ex["context"],
                    max_length=384, truncation=True, stride=128,
                    return_overflowing_tokens=True, return_offsets_mapping=True,
                    padding="max_length", return_tensors="pt",
                )
                offsets = enc_out.pop("offset_mapping")
                enc_out.pop("overflow_to_sample_mapping", None)

                # Process one chunk at a time (AOTI has fixed batch dim)
                n_chunks = enc_out["input_ids"].shape[0]
                best_s, best_e, best_score, best_off = 0, 0, -float("inf"), offsets[0]
                for ci in range(n_chunks):
                    inp = {"input_ids": enc_out["input_ids"][ci:ci+1],
                           "attention_mask": enc_out["attention_mask"][ci:ci+1]}
                    out = forward(model, inp, run_autocast, use_aoti=is_aoti)

                    if is_aoti:
                        s = out[0][0].argmax().item()
                        e_idx = out[1][0].argmax().item()
                        score = out[0][0][s].item() + out[1][0][e_idx].item()
                    else:
                        s = out.start_logits[0].argmax().item()
                        e_idx = out.end_logits[0].argmax().item()
                        score = out.start_logits[0][s].item() + out.end_logits[0][e_idx].item()

                    if score > best_score:
                        best_s, best_e = s, max(e_idx, s)
                        best_score = score
                        best_off = offsets[ci]

                s, e, off = best_s, best_e, best_off
                if s < len(off) and e < len(off):
                    predicted = ex["context"][off[s][0].item(): off[e][1].item()]
                else:
                    predicted = ""

                preds.append({"id": ex["id"], "prediction_text": predicted})
                refs.append({"id": ex["id"], "answers": ex["answers"]})

            total_f1 = total_em = 0.0
            for p, r in zip(preds, refs):
                f1, em = _squad_f1_em(p["prediction_text"], r["answers"]["text"])
                total_f1 += f1
                total_em += em
            f1 = total_f1 / len(preds) * 100
            em = total_em / len(preds) * 100
            print(f"    F1 = {f1:.2f}   EM = {em:.2f}", flush=True)
            results[label] = {"f1": round(f1, 3), "exact_match": round(em, 3)}

        except Exception as exc:
            print(f"    ERROR: {exc}", flush=True)
            results[label] = {"error": str(exc)}

    return results


# ---------------------------------------------------------------------------
# MultiNLI evaluation
# ---------------------------------------------------------------------------

_NLI_NAME2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}


def _build_label_remap(model):
    """Map model output indices to multi_nli label ints."""
    l2id = getattr(model.config, "label2id", None)
    n = getattr(model.config, "num_labels", 3)
    remap = list(range(n))
    if not l2id:
        return remap
    l2id_norm = {k.strip().lower(): int(v) for k, v in l2id.items()}
    for name, mnli_id in _NLI_NAME2ID.items():
        model_idx = l2id_norm.get(name)
        if model_idx is not None and model_idx < n:
            remap[model_idx] = mnli_id
    return remap


def run_mnli(args, quant_modes):
    """Evaluate accuracy on MultiNLI matched-validation set."""
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **_kw): return x

    print(f"\n{'─'*60}")
    print(f" MultiNLI  model={args.model}")
    print(f"{'─'*60}")

    tok = AutoTokenizer.from_pretrained(args.model)
    val_ds = load_dataset("multi_nli", split="validation_matched")
    n = resolve_num_samples(args.num_samples)
    val_ds = val_ds if n is None else val_ds.select(range(min(n, len(val_ds))))
    calib_ds = load_dataset("multi_nli", split="train").select(range(args.num_calib))

    def _encode_nli(premise, hypothesis):
        return tok(premise, hypothesis, max_length=128, truncation=True,
                   padding="max_length", return_tensors="pt")

    results = {}
    for quant_mode, alpha, run_autocast, label in iter_runs(quant_modes, args.alpha):
        print(f"\n  [{label}]", flush=True)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model, torch_dtype=torch.float32)
            model.eval()
            remap = _build_label_remap(model)

            if quant_mode != "none":
                calib = []
                for ex in calib_ds:
                    e = _encode_nli(ex["premise"], ex["hypothesis"])
                    calib.append({"input_ids": e["input_ids"],
                                  "attention_mask": e["attention_mask"]})
                model = apply_smoothquant_accuracy(model, quant_mode, alpha, calib,
                                                   use_autocast=run_autocast)

            # Optionally compile
            sample = _encode_nli("sample premise", "sample hypothesis")
            sample_dict = {"input_ids": sample["input_ids"],
                           "attention_mask": sample["attention_mask"]}
            model, is_aoti = maybe_compile_or_aoti(
                model, args, sample_dict, run_autocast, f"mnli_{label}")

            correct = total = 0
            for ex in tqdm(val_ds, desc=label, leave=False):
                e = _encode_nli(ex["premise"], ex["hypothesis"])
                inp = {"input_ids": e["input_ids"],
                       "attention_mask": e["attention_mask"]}
                out = forward(model, inp, run_autocast, use_aoti=is_aoti)
                if is_aoti:
                    raw = out[0].argmax(-1).item()
                else:
                    raw = out.logits.argmax(-1).item()
                if remap[raw] == ex["label"]:
                    correct += 1
                total += 1

            acc = correct / total * 100
            print(f"    Accuracy = {acc:.2f}%", flush=True)
            results[label] = {"accuracy": round(acc, 3)}

        except Exception as exc:
            print(f"    ERROR: {exc}", flush=True)
            results[label] = {"error": str(exc)}

    return results


# ---------------------------------------------------------------------------
# Report: read result JSONs → Markdown / CSV tables
# ---------------------------------------------------------------------------

def _md_table(headers, rows):
    col_w = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
             for i, h in enumerate(headers)]
    sep   = "| " + " | ".join("-" * w for w in col_w) + " |"
    hdr   = "| " + " | ".join(str(h).ljust(col_w[i]) for i, h in enumerate(headers)) + " |"
    lines = [hdr, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row[i]).ljust(col_w[i]) for i in range(len(headers))) + " |")
    return lines


def _csv_lines(headers, rows):
    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(headers)
    for row in rows:
        w.writerow(row)
    return buf.getvalue().splitlines()


def run_report(args):
    """Scan results dir for JSONs and print comparison tables."""
    import glob

    results_dir = args.results_dir
    json_files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    if not json_files:
        print(f"[report] No JSON files found in '{results_dir}'.", file=sys.stderr)
        return

    perf_data, squad_data, mnli_data, model_names = {}, {}, {}, {}

    for path in json_files:
        fname = os.path.basename(path)
        if "profiler" in fname:
            continue
        try:
            with open(path) as fh:
                d = json.load(fh)
        except Exception:
            continue

        model_id = d.get("model", fname)
        base = fname.replace(".json", "")
        for prefix in ("perf_", "squad_", "mnli_"):
            if base.startswith(prefix):
                base = base[len(prefix):]
                break
        model_key = base
        model_names[model_key] = model_id

        for task, tdata in d.get("tasks", {}).items():
            if task == "perf":
                perf_data.setdefault(model_key, {})
                for lbl, vals in tdata.items():
                    if "qps" in vals:
                        perf_data[model_key][lbl] = vals["qps"]
            elif task == "squad":
                squad_data.setdefault(model_key, {})
                squad_data[model_key].update(tdata)
            elif task == "mnli":
                mnli_data.setdefault(model_key, {})
                mnli_data[model_key].update(tdata)

    def _ordered_modes(data_dict):
        seen, order, extras = {}, ["none", "fp32", "bf16", "amp"], []
        for modes in data_dict.values():
            for lbl in modes:
                if lbl not in seen:
                    seen[lbl] = True
                    if lbl not in order:
                        extras.append(lbl)
        def _key(lbl):
            if lbl.startswith("smooth-static-autocast"): return (3, lbl)
            if lbl.startswith("smooth-static"): return (2, lbl)
            if lbl.startswith("smooth-dynamic-autocast"): return (5, lbl)
            if lbl.startswith("smooth-dynamic"): return (4, lbl)
            return (0, lbl)
        return [l for l in order if l in seen] + sorted(extras, key=_key)

    model_order = sorted(perf_data.keys() | squad_data.keys() | mnli_data.keys())
    sections_md, sections_csv = [], []

    if perf_data:
        modes = _ordered_modes(perf_data)
        headers = ["Model"] + modes
        rows = [[mk] + [f"{perf_data[mk].get(m, 0):.2f}" if perf_data.get(mk, {}).get(m) else "—"
                         for m in modes]
                for mk in model_order if mk in perf_data]
        sections_md.append("## Performance — QPS\n")
        sections_md.extend(_md_table(headers, rows))
        sections_md.append("")
        sections_csv.append("### Performance QPS")
        sections_csv.extend(_csv_lines(headers, rows))
        sections_csv.append("")

    for metric, mlabel in [("f1", "F1"), ("exact_match", "EM")]:
        if squad_data:
            modes = _ordered_modes(squad_data)
            headers = ["Model"] + modes
            rows = [[mk] + [f"{squad_data[mk].get(m, {}).get(metric, 0):.2f}"
                             if squad_data.get(mk, {}).get(m, {}).get(metric) is not None else "—"
                             for m in modes]
                    for mk in model_order if mk in squad_data]
            sections_md.append(f"## SQuAD 1.1 — {mlabel}\n")
            sections_md.extend(_md_table(headers, rows))
            sections_md.append("")
            sections_csv.append(f"### SQuAD {mlabel}")
            sections_csv.extend(_csv_lines(headers, rows))
            sections_csv.append("")

    if mnli_data:
        modes = _ordered_modes(mnli_data)
        headers = ["Model"] + modes
        rows = [[mk] + [f"{mnli_data[mk].get(m, {}).get('accuracy', 0):.2f}"
                         if mnli_data.get(mk, {}).get(m, {}).get("accuracy") is not None else "—"
                         for m in modes]
                for mk in model_order if mk in mnli_data]
        sections_md.append("## MultiNLI — Accuracy (%)\n")
        sections_md.extend(_md_table(headers, rows))
        sections_md.append("")
        sections_csv.append("### MultiNLI Accuracy")
        sections_csv.extend(_csv_lines(headers, rows))
        sections_csv.append("")

    emit_md = args.report_format in ("markdown", "both")
    emit_csv = args.report_format in ("csv", "both")

    if args.report_output:
        base = args.report_output
        if emit_md:
            md_path = base if base.endswith(".md") else base + ".md"
            with open(md_path, "w") as fh:
                fh.write("\n".join(sections_md) + "\n")
            print(f"Markdown report: {md_path}")
        if emit_csv:
            csv_path = base if base.endswith(".csv") else base + ".csv"
            with open(csv_path, "w") as fh:
                fh.write("\n".join(sections_csv) + "\n")
            print(f"CSV report: {csv_path}")
    else:
        if emit_md:
            print("\n".join(sections_md))
        if emit_csv:
            if emit_md:
                print("\n--- CSV ---\n")
            print("\n".join(sections_csv))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Inject report-specific args if --task report is used
    # (we parse first to check)
    pre_p = argparse.ArgumentParser(add_help=False)
    pre_p.add_argument("--task", default="squad")
    pre_args, _ = pre_p.parse_known_args()

    if pre_args.task == "report":
        p = argparse.ArgumentParser(description="Generate report from results")
        p.add_argument("--task", default="report")
        p.add_argument("--results-dir", default="results", metavar="DIR")
        p.add_argument("--report-format", default="both",
                       choices=["markdown", "csv", "both"])
        p.add_argument("--report-output", default=None, metavar="FILE")
        rargs = p.parse_args()
        run_report(rargs)
        return

    args = parse_args()

    if args.compile and args.aoti:
        print("ERROR: --compile and --aoti are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    quant_modes = ALL_QUANT_MODES if args.quant_mode == "all" else [args.quant_mode]
    tasks = ["squad", "mnli"] if args.task == "all" else [args.task]

    all_results = {"model": args.model, "tasks": {}}

    for task in tasks:
        if task == "squad":
            all_results["tasks"]["squad"] = run_squad(args, quant_modes)
        elif task == "mnli":
            all_results["tasks"]["mnli"] = run_mnli(args, quant_modes)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps(all_results, indent=2))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"\nResults written to: {args.output}")

    # Print best-alpha CSV summary when multiple alphas were tested
    if len(args.alpha) > 1:
        csv_path = None
        if args.output:
            csv_path = args.output.replace(".json", "") + "_best.csv"
        _print_best_alpha_summary(all_results, csv_path)


# ---------------------------------------------------------------------------
# Best-alpha summary
# ---------------------------------------------------------------------------

_ALPHA_RE = re.compile(r"^(smooth-[\w-]+?)\(a=([\d.]+)\)$")


def _extract_metric(entry, task):
    """Return the primary metric value from a result entry, or None."""
    if "error" in entry:
        return None
    if task == "squad":
        return entry.get("f1")
    elif task == "mnli":
        return entry.get("accuracy")
    return None


def _print_best_alpha_summary(all_results, csv_path=None):
    """Scan results for multiple alpha runs and print a CSV of the best."""
    import csv, io

    model_id = all_results.get("model", "unknown")
    rows = []  # (task, quant_mode, best_alpha, metric_name, metric_value)

    for task, task_results in all_results.get("tasks", {}).items():
        metric_name = "F1" if task == "squad" else "Accuracy"

        # Baselines (fp32 and/or bf16)
        for bl_label in ("fp32", "bf16", "none"):
            baseline = task_results.get(bl_label)
            if baseline is not None:
                val = _extract_metric(baseline, task)
                if val is not None:
                    rows.append((task, bl_label, "", metric_name, f"{val:.2f}"))

        # Group by quant mode, find best alpha
        mode_best = {}  # mode -> (best_val, best_alpha, all_entries)
        for label, entry in task_results.items():
            m = _ALPHA_RE.match(label)
            if not m:
                continue
            mode, alpha_str = m.group(1), m.group(2)
            val = _extract_metric(entry, task)
            if val is None:
                continue
            prev = mode_best.get(mode)
            if prev is None or val > prev[0]:
                mode_best[mode] = (val, alpha_str, entry)

        for mode in ("smooth-static", "smooth-static-autocast",
                     "smooth-dynamic", "smooth-dynamic-autocast"):
            if mode in mode_best:
                best_val, best_alpha, _ = mode_best[mode]
                rows.append((task, mode, best_alpha, metric_name, f"{best_val:.2f}"))

    if not rows:
        return

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["model", "task", "quant_mode", "best_alpha", "metric", "value"])
    for task, mode, alpha, metric_name, val in rows:
        w.writerow([model_id, task, mode, alpha, metric_name, val])
    csv_text = buf.getvalue()

    print(f"\n{'='*60}")
    print("BEST ALPHA SUMMARY (CSV)")
    print("=" * 60)
    print(csv_text, end="")

    if csv_path:
        with open(csv_path, "w") as fh:
            fh.write(csv_text)
        print(f"\nBest-alpha CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
