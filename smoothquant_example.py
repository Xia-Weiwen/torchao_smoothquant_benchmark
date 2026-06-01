import argparse
from transformers import AutoTokenizer, AutoModel
import os
import torch
import time
import multiprocessing as mp

from smoothquant_utils import (
    apply_smoothquant, optional_autocast, do_compile, do_aoti_compile, infer,
    inductor_config as config,
)


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--model",
    type=str,
    default="xlm-roberta-base",
    choices=["xlm-roberta-base", "bert-large-uncased", "distilbert-base-uncased"],
)
argparser.add_argument("--quant-mode", type=str, default="none", choices=["smooth-static", "smooth-dynamic", "none"])
argparser.add_argument("--autocast", action='store_true')
argparser.add_argument("--aoti", action='store_true', help="Use AOTInductor.")
argparser.add_argument("--warmup", type=int, default=20)
argparser.add_argument("--active", type=int, default=50)
argparser.add_argument("--profile", action='store_true')
argparser.add_argument("--cores-per-instance", type=int, default=4)
argparser.add_argument("--cpu-name", type=str, default="", help="CPU label to record in result.csv")
args = argparser.parse_args()


if args.profile:
    config.cpp.enable_kernel_profile = True


available_cores = list(os.sched_getaffinity(0))
num_cores = len(available_cores)
num_instances = num_cores // args.cores_per_instance


log_path = "./benchmark_logs"
if not os.path.exists(log_path):
    os.makedirs(log_path)


AUTOCAST = args.autocast
encoder = args.model
tokenizer = AutoTokenizer.from_pretrained(encoder)
sentence_256 = "The rapid advancement of artificial intelligence and machine learning technologies has fundamentally transformed the landscape of modern computing and business operations. Companies across various industries are increasingly investing in cloud infrastructure, data analytics platforms, and automated systems to enhance their competitive advantage and operational efficiency. These technological innovations have enabled organizations to process vast amounts of information in real-time, make data-driven decisions with unprecedented accuracy, and deliver personalized experiences to their customers at scale. However, the implementation of such sophisticated systems requires careful consideration of multiple factors including data privacy, security protocols, regulatory compliance, and ethical implications. Furthermore, the integration of these technologies demands significant organizational changes, including workforce training and development, process reengineering, and cultural adaptation to embrace digital transformation. Industry leaders emphasize that successful technology adoption goes beyond merely deploying new tools and platforms; it requires a holistic approach that aligns technology strategy with business objectives, fosters innovation culture, and creates sustainable long-term value for all key stakeholders while ensuring competitiveness."
if args.model in ("bert-large-uncased", "distilbert-base-uncased"):
    # BERT/DistilBERT WordPiece tokenizes the same passage to fewer tokens than RoBERTa's
    # SentencePiece, so pad to a fixed 256 to keep input length consistent.
    query = tokenizer(
        sentence_256,
        return_tensors='pt',
        padding='max_length',
        max_length=256,
        truncation=True,
    )
else:
    query = tokenizer(sentence_256, return_tensors='pt', padding=True)
input_ids, attention_mask = query['input_ids'], query['attention_mask']
print("[info] input_ids shape:", input_ids.shape)
model_inputs = (input_ids, attention_mask)

model = AutoModel.from_pretrained(encoder)
model.eval()

if args.quant_mode != "none":
    model = apply_smoothquant(model, args.quant_mode, 0.5, model_inputs,
                              use_autocast=AUTOCAST, filter_fn=None)


# Bind autocast helpers to the global AUTOCAST setting
_infer = optional_autocast(lambda m, inp: m(*inp), AUTOCAST)
_compile = lambda m: do_compile(m, AUTOCAST)

def _aot_compile(m):
    save_path = os.path.join(os.getcwd(), f"{args.model.replace('/', '_')}_quant_{args.quant_mode}_autocast_{args.autocast}.pt2")
    return do_aoti_compile(m, model_inputs, AUTOCAST, save_path)


KMP_AFFINITY = os.environ.get('KMP_AFFINITY', '')
os.environ['KMP_AFFINITY'] = f'granularity=fine,compact,1,{available_cores[0]}'
torch.set_num_threads(args.cores_per_instance)
os.environ['OMP_NUM_THREADS'] = f'{args.cores_per_instance}'
os.sched_setaffinity(os.getpid(), available_cores[:args.cores_per_instance])
torch._dynamo.reset() # reset dynamo to clear the cache and profile the compilation
compiled_model = _aot_compile(model) if args.aoti else _compile(model)
# run once so that torch.compile takes effect
_infer(compiled_model, model_inputs)
os.environ['KMP_AFFINITY'] = KMP_AFFINITY


def get_log_file_path(inst_id):
    safe_model = args.model.replace('/', '_')
    return os.path.join(log_path, f"log_{safe_model}_{args.quant_mode}_autocast_{args.autocast}_inst_{inst_id}.txt")

def run_benchmark(compiled_model, model_inputs, inst_id, core_list):
    log_file_path = get_log_file_path(inst_id)
    os.sched_setaffinity(os.getpid(), core_list)
    torch.set_num_threads(len(core_list))
    os.environ['OMP_NUM_THREADS'] = f'{len(core_list)}'
    for _ in range(args.warmup):
        _infer(compiled_model, model_inputs)
    t0 = time.time()
    for _ in range(args.active):
        _infer(compiled_model, model_inputs)
    elapsed = time.time() - t0
    rps = args.active / elapsed # requests per second
    qlinear_per_iter_ms = None  # filled in below if profiling

    if args.profile:
        import re
        from collections import defaultdict
        mnk_re = re.compile(r"m(\d+)_?n(\d+)_?k(\d+)")
        profile_active = 6  # must match schedule(active=...) below
        qlinear_totals_us = []  # captured by trace_handler

        def summarize_qlinear(prof, out_file):
            """Group qlinear / GEMM events by MNK and print/write a summary."""
            by_shape = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
            unshaped = defaultdict(lambda: [0.0, 0])
            for evt in prof.key_averages():
                name = evt.key
                total_us = float(evt.self_cpu_time_total)
                count = int(evt.count)
                m = mnk_re.search(name)
                if m:
                    shape = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                    kind = 'qlinear_kernel' if 'qlinear' in name.lower() else 'gemm_kernel'
                    by_shape[shape][kind][0] += total_us
                    by_shape[shape][kind][1] += count
                elif 'qlinear' in name.lower():
                    unshaped[name][0] += total_us
                    unshaped[name][1] += count
                elif name.startswith('aoti_torch_cpu__linear_pointwise'):
                    unshaped[name][0] += total_us
                    unshaped[name][1] += count

            total_qlinear_us = 0.0
            lines = []
            lines.append("=" * 105)
            lines.append(f"QLINEAR / GEMM SUMMARY - Instance {inst_id} (grouped by M x N x K)")
            lines.append("=" * 105)
            lines.append(f"{'Shape (MxNxK)':<25} {'Kind':<20} {'Self CPU (ms)':>15} {'# Calls':>10} {'avg/call (us)':>15} {'avg/iter (us)':>15}")
            lines.append("-" * 105)
            for shape in sorted(by_shape.keys()):
                shape_str = f"{shape[0]}x{shape[1]}x{shape[2]}"
                for kind, (t_us, c) in sorted(by_shape[shape].items()):
                    avg_call = t_us / c if c else 0
                    avg_iter = t_us / profile_active
                    lines.append(f"{shape_str:<25} {kind:<20} {t_us/1000:>15.3f} {c:>10} {avg_call:>15.2f} {avg_iter:>15.2f}")
                    total_qlinear_us += t_us
            if unshaped:
                lines.append("-" * 105)
                lines.append("Unshaped (dispatch / no MNK tag):")
                for name, (t_us, c) in sorted(unshaped.items(), key=lambda x: -x[1][0]):
                    avg_call = t_us / c if c else 0
                    avg_iter = t_us / profile_active
                    short = name if len(name) <= 50 else name[:47] + '...'
                    lines.append(f"  {short:<50} {t_us/1000:>12.3f} ms  calls={c:<6} avg/call={avg_call:.2f} us  avg/iter={avg_iter:.2f} us")
                    total_qlinear_us += t_us
            lines.append("-" * 105)
            lines.append(
                f"Qlinear/GEMM Self CPU time: {total_qlinear_us/1000:.3f} ms "
                f"(avg {total_qlinear_us/1000/profile_active:.3f} ms/iter over {profile_active} active iters)"
            )
            text = "\n".join(lines)
            if inst_id == 0:
                print(text)
            out_file.write("\n" + text + "\n")
            qlinear_totals_us.append(total_qlinear_us)
            return total_qlinear_us

        def trace_handler(prof):
            table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1, max_name_column_width=300)
            with open(log_file_path, "w") as f:
                f.write(table)
                summarize_qlinear(prof, f)
            if inst_id == 0:
                print(table)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=profile_active),
            on_trace_ready=trace_handler,
        ) as prof:
            for _ in range(2 + 2 + profile_active):
                _infer(compiled_model, model_inputs)
                prof.step()

        if qlinear_totals_us:
            qlinear_per_iter_ms = (sum(qlinear_totals_us) / len(qlinear_totals_us)) / 1000 / profile_active

    with open(log_file_path, "a") as f:
        f.write(f"\nModel: {args.model}, input length: {input_ids.shape[1]}, quantization mode: {args.quant_mode}, autocast: {args.autocast}, aoti: {args.aoti}\n")
        f.write(f"RPS: {round(rps, 2)}\n")
        if qlinear_per_iter_ms is not None:
            f.write(f"QLINEAR_PER_ITER_MS: {qlinear_per_iter_ms:.6f}\n")


processes = []
KMP_AFFINITY = os.environ.get('KMP_AFFINITY', '')
for i in range(0, num_instances):
    cores = [available_cores[j] for j in range(i * args.cores_per_instance, (i + 1) * args.cores_per_instance)]
    os.environ['KMP_AFFINITY'] = f'granularity=fine,compact,1,{cores[0]}'
    p = mp.Process(
        target=run_benchmark, args=(compiled_model, model_inputs, i, cores)
    )
    processes.append(p)
    p.start()
os.environ['KMP_AFFINITY'] = KMP_AFFINITY
for p in processes:
    p.join()

# Get RPS (and qlinear time if profiled) from all log files and put together
rps_list = []
qlinear_per_iter_ms_list = []
for i in range(num_instances):
    log_file_path = get_log_file_path(i)
    with open(log_file_path, "r") as f:
        lines = f.readlines()
        rps_line = [line for line in lines if line.startswith("RPS:")][0]
        rps = float(rps_line.strip().split("RPS:")[1])
        rps_list.append(rps)
        ql_lines = [line for line in lines if line.startswith("QLINEAR_PER_ITER_MS:")]
        if ql_lines:
            qlinear_per_iter_ms_list.append(float(ql_lines[0].strip().split(":", 1)[1]))
total_rps = sum(rps_list)
print("=" * 20)
avg_qlinear_ms = ""
if qlinear_per_iter_ms_list:
    n = len(qlinear_per_iter_ms_list)
    total_q = sum(qlinear_per_iter_ms_list)
    avg_qlinear_ms = total_q / n

# Append a row to result.csv for easy aggregation across runs.
import csv
csv_path = "result.csv"
csv_fields = [
    "cpu_name", "num_cores", "num_instances", "model", "quant_mode",
    "autocast", "aoti", "total_rps", "avg_rps", "avg_qlinear_ms",
]
csv_row = {
    "cpu_name": args.cpu_name,
    "num_cores": num_cores,
    "num_instances": num_instances,
    "model": args.model,
    "quant_mode": args.quant_mode,
    "autocast": args.autocast,
    "aoti": args.aoti,
    "total_rps": round(total_rps, 2),
    "avg_rps": round(total_rps / num_instances, 2),
    "avg_qlinear_ms": (round(avg_qlinear_ms, 3) if avg_qlinear_ms != "" else ""),
}
write_header = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    if write_header:
        writer.writeheader()
    writer.writerow(csv_row)

# Also echo the CSV row to stdout for quick inspection.
import io
_buf = io.StringIO()
_w = csv.DictWriter(_buf, fieldnames=csv_fields)
_w.writeheader()
_w.writerow(csv_row)
print(_buf.getvalue(), end="")
print("=" * 20)