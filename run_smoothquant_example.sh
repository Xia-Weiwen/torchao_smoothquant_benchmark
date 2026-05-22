export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:${CONDA_PREFIX}/lib/libtcmalloc.so
export ONEDNN_CACHE_CONTEXT_UNSAFE=1

CPU_NAME=GNR

rm -rf /tmp/torchinductor_`whoami`/* # clear Inductor cache
n_cores=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)
max_idx=$(( $n_cores - 1 ))
taskset -c 0-${max_idx} python smoothquant_example.py --cpu-name ${CPU_NAME} --model xlm-roberta-base --quant-mode none --aoti --profile
taskset -c 0-${max_idx} python smoothquant_example.py --cpu-name ${CPU_NAME} --model xlm-roberta-base --quant-mode none --autocast --aoti --profile
taskset -c 0-${max_idx} python smoothquant_example.py --cpu-name ${CPU_NAME} --model xlm-roberta-base --quant-mode smooth-dynamic --autocast --aoti --profile
taskset -c 0-${max_idx} python smoothquant_example.py --cpu-name ${CPU_NAME} --model xlm-roberta-base --quant-mode smooth-static --autocast --aoti --profile
taskset -c 0-${max_idx} python smoothquant_example.py --cpu-name ${CPU_NAME} --model bert-large-uncased --quant-mode none --aoti --profile
taskset -c 0-${max_idx} python smoothquant_example.py --cpu-name ${CPU_NAME} --model bert-large-uncased --quant-mode none --autocast --aoti --profile
taskset -c 0-${max_idx} python smoothquant_example.py --cpu-name ${CPU_NAME} --model bert-large-uncased --quant-mode smooth-dynamic --autocast --aoti --profile
taskset -c 0-${max_idx} python smoothquant_example.py --cpu-name ${CPU_NAME} --model bert-large-uncased --quant-mode smooth-static --autocast --aoti --profile
taskset -c 0-${max_idx} python smoothquant_example.py --cpu-name ${CPU_NAME} --model distilbert-base-uncased --quant-mode none --aoti --profile
taskset -c 0-${max_idx} python smoothquant_example.py --cpu-name ${CPU_NAME} --model distilbert-base-uncased --quant-mode none --autocast --aoti --profile
taskset -c 0-${max_idx} python smoothquant_example.py --cpu-name ${CPU_NAME} --model distilbert-base-uncased --quant-mode smooth-dynamic --autocast --aoti --profile
taskset -c 0-${max_idx} python smoothquant_example.py --cpu-name ${CPU_NAME} --model distilbert-base-uncased --quant-mode smooth-static --autocast --aoti --profile