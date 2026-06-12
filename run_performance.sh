export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:${CONDA_PREFIX}/lib/libtcmalloc.so
export ONEDNN_CACHE_CONTEXT_UNSAFE=1

CPU_NAME=$(lscpu | grep "Model name" | head -n 1 | awk -F: '{print $2}' | xargs)
echo "Running on CPU: ${CPU_NAME}"

rm -rf /tmp/torchinductor_`whoami`/* # clear Inductor cache
n_cores=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)
max_idx=$(( $n_cores - 1 ))
model_list="xlm-roberta-base bert-large-uncased distilbert-base-uncased"
extra_args="--aoti --profile"
cases=(
    "--quant-mode none"
    "--quant-mode none --autocast"
    "--quant-mode smooth-dynamic --autocast"
    "--quant-mode smooth-static --autocast"
    # "--quant-mode smooth-dynamic" # uncomment these lines to run without autocast
    # "--quant-mode smooth-static"
)
for model in ${model_list}; do
    for case in "${cases[@]}"; do
        echo "Running ${model} with case: ${case}"
        taskset -c 0-${max_idx} python benchmark_performance.py --cpu-name "${CPU_NAME}" --model ${model} ${case} ${extra_args}
    done
done
