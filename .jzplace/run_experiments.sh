#!/bin/bash
# ps -ef | grep python | grep -v grep | awk '{print $2}' | sudo xargs kill -9
# ps -ef | grep sglang | grep -v grep | awk '{print $2}' | sudo xargs kill -9

set -x
model="meta-llama/Llama-2-7b-chat-hf"
speculative_draft="lmzheng/sglang-EAGLE-llama2-chat-7B"
results_dir="mab_results/llama2-7b-plots"
tp_size=1
mkdir -p $results_dir
port_start=8082

gpu_index=0
for traffic_rate_option in "concurrency"; do
    for temperature in 0 5; do
        for sharegpt_context_len in 256 4096; do
            port=$(($port_start + $gpu_index))

            # Run the command and show the output both via command line and results to mab_results/testing_$traffic_rate_option\_$temperature.log
            CUDA_VISIBLE_DEVICES=$gpu_index python .jzplace/eagle_testing.py \
                --traffic-rate-option="$traffic_rate_option" \
                --traffic-rate-list=1,4,16,64,256 \
                --sharegpt-context-len="$sharegpt_context_len" \
                --temperature="$temperature" \
                --port="$port" \
                --results-dir="$results_dir" \
                --model="$model" \
                --tp-size="$tp_size" \
                --speculative-draft="$speculative_draft" \
            > ${results_dir}/testing_${traffic_rate_option}_${temperature}_${sharegpt_context_len}.log 2>&1 &
            gpu_index=$((gpu_index + 1))
        done
    done
done
