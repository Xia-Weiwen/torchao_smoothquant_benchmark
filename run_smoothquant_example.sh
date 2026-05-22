export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:${CONDA_PREFIX}/lib/libtcmalloc.so
export ONEDNN_CACHE_CONTEXT_UNSAFE=1

rm -rf /tmp/torchinductor_`whoami`/* # clear Inductor cache
n_cores=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)
max_idx=$(( $n_cores - 1 ))
script=smoothquant_example.py
model=xlm-roberta-base
log_dir="./benchmark_logs_$(date +%Y%m%d_%H%M%S)" # new log dir with date time
taskset -c 0-${max_idx} python $script --model $model --quant-mode none --aoti --profile --log-dir $log_dir
taskset -c 0-${max_idx} python $script --model $model --quant-mode none --autocast --aoti --profile --log-dir $log_dir
taskset -c 0-${max_idx} python $script --model $model --quant-mode smooth-dynamic --autocast --aoti --profile --log-dir $log_dir
taskset -c 0-${max_idx} python $script --model $model --quant-mode smooth-static --autocast --aoti --profile --log-dir $log_dir
