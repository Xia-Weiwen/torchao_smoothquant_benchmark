import argparse
from transformers import AutoTokenizer, AutoModel
import os
import torch
import time
import torchao
# must import this to register the fusion passes
import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer
from torchao.prototype.smoothquant import SmoothQuantConfig
from torchao.quantization.quantize_.common.quantization_step import QuantizationStep
from torchao.quantization.granularity import PerTensor, PerRow
from torchao.quantization.quant_api import (
    Int8DynamicActivationInt8WeightConfig,
    Int8StaticActivationInt8WeightConfig,
)
import multiprocessing as mp

import torch._inductor.config as config
config.freezing = True
config.max_autotune = True
config.cpp_wrapper = True


argparser = argparse.ArgumentParser()
argparser.add_argument("--model", type=str, default="xlm-roberta-base")
argparser.add_argument("--quant-mode", type=str, default="none", choices=["smooth-static", "smooth-dynamic", "none"])
argparser.add_argument("--autocast", action='store_true')
argparser.add_argument("--aoti", action='store_true', help="Use AOTInductor.")
argparser.add_argument("--warmup", type=int, default=20)
argparser.add_argument("--active", type=int, default=50)
argparser.add_argument("--profile", action='store_true')
argparser.add_argument("--cores-per-instance", type=int, default=4)
argparser.add_argument("--log-dir", type=str, default="./benchmark_logs")
args = argparser.parse_args()


if args.profile:
    config.cpp.enable_kernel_profile = True


available_cores = list(os.sched_getaffinity(0))
num_cores = len(available_cores)
num_instances = num_cores // args.cores_per_instance


log_path = args.log_dir
if not os.path.exists(log_path):
    os.makedirs(log_path)


print("=" * 20)
print(f"Running model: {args.model}, quantization mode: {args.quant_mode}, autocast: {args.autocast}, aoti: {args.aoti}",
      f"total cores: {num_cores}, cores per instance: {args.cores_per_instance}, num instances: {num_instances}")
print("=" * 20)


AUTOCAST = args.autocast
encoder = args.model
tokenizer = AutoTokenizer.from_pretrained(encoder)
sentence_256 = "The rapid advancement of artificial intelligence and machine learning technologies has fundamentally transformed the landscape of modern computing and business operations. Companies across various industries are increasingly investing in cloud infrastructure, data analytics platforms, and automated systems to enhance their competitive advantage and operational efficiency. These technological innovations have enabled organizations to process vast amounts of information in real-time, make data-driven decisions with unprecedented accuracy, and deliver personalized experiences to their customers at scale. However, the implementation of such sophisticated systems requires careful consideration of multiple factors including data privacy, security protocols, regulatory compliance, and ethical implications. Furthermore, the integration of these technologies demands significant organizational changes, including workforce training and development, process reengineering, and cultural adaptation to embrace digital transformation. Industry leaders emphasize that successful technology adoption goes beyond merely deploying new tools and platforms; it requires a holistic approach that aligns technology strategy with business objectives, fosters innovation culture, and creates sustainable long-term value for all key stakeholders while ensuring competitiveness."
query = tokenizer(sentence_256, return_tensors='pt', padding=True)
input_ids, attention_mask = query['input_ids'], query['attention_mask']
print("[info] input_ids shape:", input_ids.shape)
model_inputs = (input_ids, attention_mask)

model = AutoModel.from_pretrained(encoder)
model.eval()

if args.quant_mode != "none":
    if args.quant_mode == "smooth-dynamic": # dynamic smooth quant
        base_config=Int8DynamicActivationInt8WeightConfig(
            version=2,
            granularity=[PerRow(),PerRow()],
        )
    elif args.quant_mode == "smooth-static": # static smooth quant
        base_config=Int8StaticActivationInt8WeightConfig(
            granularity=[PerTensor(),PerRow()],
        )

    quant_config = SmoothQuantConfig(
        base_config=base_config,
        step=QuantizationStep.PREPARE,
        alpha=0.5,
    )

    torchao.quantization.quantize_(model, quant_config)
    model(*model_inputs)
    quant_config.step = QuantizationStep.CONVERT
    torchao.quantization.quantize_(model, quant_config)


def optional_autocast(func): # decorator
    def f(*args, **kw):
        if AUTOCAST:
            with torch.no_grad(), torch.autocast('cpu'):
                rv = func(*args, **kw)
        else:
            with torch.no_grad():
                rv = func(*args, **kw)
        return rv
    return f


@optional_autocast
def infer(model, model_inputs):
    return model(*model_inputs)


@optional_autocast
def compile(model):
    options={"guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe}
    return torch.compile(model, options=options, fullgraph=True)


@optional_autocast
def aot_compile(model):
    import torch._export.utils as eu
    with eu._disable_aten_to_metadata_assertions():
        exported = torch.export.export(model, args=model_inputs)
    save_path = os.path.join(os.getcwd(), f"{args.model}_quant_{args.quant_mode}_autocast_{args.autocast}.pt2")
    output_path = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=save_path,
    )
    return torch._inductor.aoti_load_package(save_path)


KMP_AFFINITY = os.environ.get('KMP_AFFINITY', '')
os.environ['KMP_AFFINITY'] = f'granularity=fine,compact,1,{available_cores[0]}'
torch.set_num_threads(args.cores_per_instance)
os.environ['OMP_NUM_THREADS'] = f'{args.cores_per_instance}'
os.sched_setaffinity(os.getpid(), available_cores[:args.cores_per_instance])
torch._dynamo.reset() # reset dynamo to clear the cache and profile the compilation
compiled_model = aot_compile(model) if args.aoti else compile(model)
# run once so that torch.compile takes effect
infer(compiled_model, model_inputs)
os.environ['KMP_AFFINITY'] = KMP_AFFINITY


def get_log_file_path(inst_id):
    return os.path.join(log_path, f"log_{args.model}_{args.quant_mode}_autocast_{args.autocast}_inst_{inst_id}.txt")

def run_benchmark(compiled_model, model_inputs, inst_id, core_list):
    log_file_path = get_log_file_path(inst_id)
    os.sched_setaffinity(os.getpid(), core_list)
    torch.set_num_threads(len(core_list))
    os.environ['OMP_NUM_THREADS'] = f'{len(core_list)}'
    for _ in range(args.warmup):
        infer(compiled_model, model_inputs)
    t0 = time.time()
    for _ in range(args.active):
        infer(compiled_model, model_inputs)
    elapsed = time.time() - t0
    rps = args.active / elapsed # requests per second

    if args.profile:

        def trace_handler(prof):
            with open(log_file_path, "w") as f:
                f.write(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
            if inst_id == 0: # print the profile result of the first instance to console
                print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=6),
            on_trace_ready=trace_handler,
        ) as prof:
            for _ in range(10):
                infer(compiled_model, model_inputs)
                prof.step()

    with open(log_file_path, "w") as f:
        f.write(f"Model: {args.model}, input length: {input_ids.shape[1]}, quantization mode: {args.quant_mode}, autocast: {args.autocast}, aoti: {args.aoti}\n")
        f.write(f"RPS: {round(rps, 2)}\n")


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

# Get RPS from all log files and put together
rps_list = []
for i in range(num_instances):
    log_file_path = get_log_file_path(i)
    with open(log_file_path, "r") as f:
        lines = f.readlines()
        rps_line = [line for line in lines if line.startswith("RPS:")][0]
        # print(i, rps_line.strip())
        rps = float(rps_line.strip().split("RPS:")[1])
        rps_list.append(rps)
total_rps = sum(rps_list)
print("=" * 20)
print(f"Model: {args.model}, quantization mode: {args.quant_mode}, autocast: {args.autocast}, aoti: {args.aoti}",
      f"total cores: {num_cores}, cores per instance: {args.cores_per_instance}, num instances: {num_instances}")
print(f"Total RPS: {round(total_rps, 2)}, average RPS per instance: {round(total_rps / num_instances, 2)}")
print(f"Logs saved to: {log_path}")
print("=" * 20)
