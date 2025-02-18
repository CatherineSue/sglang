#!/usr/bin/env python3

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

# Get current environment and ensure PYTHONPATH is preserved
ENV = os.environ.copy()

# List of experiments to run
EXPERIMENTS = [
    # MAB to adaptively choose among different speculative decoding settings 
    "EG,2_2_4,3_4_8,5_8_16",
    "EG,1_2_2,3_2_4,3_4_8,5_8_16",
    "UCB1,2_2_4,3_4_8,5_8_16",
    # No speculative decoding. Has an option to turn off overlap scheduling
    "None,disable-overlap-schedule",
    "None",
    # A single speculative decoding setting. No need of EG or UCB1 here. EG is just a placeholder in notation.
    "EG,1_1_1",
    "EG,1_2_2",
    "EG,2_2_4",
    "EG,3_2_4",
    "EG,3_4_8",
    "EG,5_8_16",
]

def kill_server(process=None, port=None):
    """Kill server process and cleanup port.
    
    Args:
        process: Optional subprocess.Popen object of the server
        port: Port number to cleanup after process termination
    """
    if process is not None:
        try:
            # Safety check: ensure we're not killing our own process group
            server_pgid = os.getpgid(process.pid)
            if server_pgid != os.getpgid(os.getpid()):
                # Kill the entire process group
                os.killpg(server_pgid, signal.SIGTERM)
                process.wait(timeout=5)
                
                # Force kill if still running
                if process.poll() is None:
                    os.killpg(server_pgid, signal.SIGKILL)
                    process.wait(timeout=1)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass  # Process or group already gone
    
    try:
        # Cleanup any remaining port binding
        subprocess.run(["fuser", "-k", f"{port}/tcp"], 
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
        time.sleep(1)  # Brief wait for port cleanup
    except:
        pass

def start_server(mab_config, port):
    """Start the server with the given configuration."""
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model=meta-llama/Llama-2-7b-chat-hf",
        "--mem-fraction=0.7",
        f"--port={port}",
        "--random-seed=42",
    ]
    
    if mab_config.startswith("None"):
        options = mab_config.split(",")[1:]
        for option in options:
            cmd.extend([f"--{option}"])

    else:
        cmd.extend([
            "--speculative-algo=EAGLE",
            "--speculative-draft=lmzheng/sglang-EAGLE-llama2-chat-7B",
            f"--speculative-eagle-mab={mab_config}"
        ])
        # Parse the first configuration after algorithm type for num_steps, topk, and draft_tokens
        parts = mab_config.split(",")
        if len(parts) >= 2:  # At least algorithm and one config
            config_parts = parts[-1].split("_")
            assert len(config_parts) == 3, "Invalid config format"
            cmd.extend([
                f"--speculative-num-steps={config_parts[0]}",
                f"--speculative-eagle-topk={config_parts[1]}",
                f"--speculative-num-draft-tokens={config_parts[2]}"
            ])
    
    # Start process in its own process group
    print(" ".join(cmd))
    process = subprocess.Popen(cmd, env=ENV, preexec_fn=os.setsid)
    while True:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                break
        except (ConnectionRefusedError, socket.timeout):
            time.sleep(5)
    print(f"Server started on port {port}")
    return process

def run_benchmark(output_file, port, num_prompts=100, traffic_rate_option='qps'):
    """Run the benchmark with specified parameters."""
    if traffic_rate_option == 'qps':
        for request_rate in [1, 4, 16, 64, 256]:
            cmd = [
                "python3", "-m", "sglang.bench_serving",
                "--backend=sglang-oai",
                "--dataset-name=sharegpt",
                f"--num-prompts={max(100, int(num_prompts * min(1, request_rate/10)))}",
                "--max-concurrency=256",
                f"--request-rate={request_rate}",
                f"--port={port}",
                f"--output-file={output_file}"
            ]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True, env=ENV)
            time.sleep(10)
    else:
        assert traffic_rate_option == 'concurrency', "Invalid options"
        for concurrency in [1, 4, 16, 64, 256]:
            cmd = [
                "python3", "-m", "sglang.bench_serving",
                "--backend=sglang-oai",
                "--dataset-name=sharegpt",
                f"--num-prompts={max(100, int(num_prompts * min(1, concurrency/10)))}",
                f"--max-concurrency={concurrency}",
                "--request-rate=256",
                f"--port={port}",
                f"--output-file={output_file}"
            ]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True, env=ENV)
            time.sleep(10)

def plot_results(EXPERIMENTS, results_dir):
    """Generate plots from the results."""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Set the figure background to white
    fig.patch.set_facecolor('white')
    for ax in axes.flat:
        ax.set_facecolor('white')
    
    # Color scheme from memory
    colors = [plt.cm.tab10(i % 10) for i in range(len(EXPERIMENTS))]
    
    for exp_idx, exp in enumerate(EXPERIMENTS):
        result_file = Path(results_dir) / f"{exp}.jsonl"
        if not result_file.exists():
            print(f"Warning: Result file not found for {exp}")
            continue
            
        try:
            data = []
            with open(result_file) as f:
                for line in f:
                    data.append(json.loads(line))
            
            if not data:
                print(f"Warning: No data found in {result_file}")
                continue
            
            # Extract metrics for plotting
            output_throughput = [d['output_throughput'] for d in data]
            input_throughput = [d['input_throughput'] for d in data]
            mean_otps = [1000/d['mean_tpot_ms'] for d in data]
            mean_ttft_ms = [d['mean_ttft_ms'] for d in data]
            total_throughput = [i + o for i, o in zip(input_throughput, output_throughput)]
            
            color = colors[exp_idx]
            
            # Plot the four subplots
            axes[0, 0].plot(output_throughput, mean_otps, 'o-', label=exp, color=color)
            axes[0, 1].plot(output_throughput, mean_ttft_ms, 'o-', label=exp, color=color)
            axes[1, 0].plot(total_throughput, mean_otps, 'o-', label=exp, color=color)
            axes[1, 1].plot(total_throughput, mean_ttft_ms, 'o-', label=exp, color=color)
            
        except Exception as e:
            print(f"Error processing {exp}: {str(e)}")
            continue
    
    # Configure the plots
    for ax in axes.flat:
        ax.grid(True)
        ax.legend()
        # ax.set_xscale('log')  # Use log scale for time metrics
        # ax.set_yscale('log')  # Use log scale for throughput
    
    axes[0, 0].set_ylabel('Mean Inference Speed')
    axes[0, 0].set_xlabel('Output Throughput')
    axes[0, 1].set_ylabel('Mean TTFT (ms)')
    axes[0, 1].set_xlabel('Output Throughput')
    axes[1, 0].set_ylabel('Mean Inference Speed')
    axes[1, 0].set_xlabel('Total Throughput')
    axes[1, 1].set_ylabel('Mean TTFT (ms)')
    axes[1, 1].set_xlabel('Total Throughput')
    
    plt.tight_layout()
    plt.savefig(Path(results_dir) / 'speculative_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=30023)
    parser.add_argument('--results-dir', type=str, default='mab_results')
    parser.add_argument('--num-prompts', type=int, default=1000)
    parser.add_argument('--traffic-rate-option', type=str, default='concurrency', choices=['qps', 'concurrency'])
    
    args = parser.parse_args()
    print(args)

    # Create results directory if it doesn't exist
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    
    for exp in EXPERIMENTS:
        print(f"\nRunning experiment: {exp}")
        
        # Kill any existing processes on port
        kill_server(port=args.port)
        
        result_file = Path(args.results_dir) / f"{exp}.jsonl"
        # if the result_file doesn't exist, create it
        if not result_file.exists():
            # Start server with current configuration
            server_process = start_server(exp, args.port)
            
            try:
                # Run warmup experiments
                if len(exp.split(',')) > 2:
                    warmup_iter = 2
                    num_prompts = args.num_prompts
                else:
                    warmup_iter = 1
                    num_prompts = min(30, args.num_prompts)

                for i in range(warmup_iter):
                    print(f"  Warmup run {i+1}/{warmup_iter}")
                    run_benchmark(f"/tmp/warmup_{i}.jsonl", args.port, num_prompts, args.traffic_rate_option)
                
                # Run final experiment
                print("  Final run")
                run_benchmark(str(result_file), args.port, args.num_prompts, args.traffic_rate_option)
                
            except Exception as e:
                print(f"Error running experiment: {str(e)}")
                kill_server(process=server_process, port=args.port)
                continue
            finally:
                # Always kill the server when done
                kill_server(process=server_process, port=args.port)
    
        # Generate plots
        print("\nGenerating plots...")
        plot_results(EXPERIMENTS, args.results_dir)

    print("Done! Results are in the results directory.")

if __name__ == '__main__':
    main()

# python /home/jingqzha/projects/sglang/.jzplace/run_speculative_experiments.py --traffic-rate-option=concurrency
# mv /home/jingqzha/projects/sglang/.jzplace/results /home/jingqzha/projects/sglang/.jzplace/results_concurrency_new

# python /home/jingqzha/projects/sglang/.jzplace/run_speculative_experiments.py --traffic-rate-option=qps
# mv /home/jingqzha/projects/sglang/.jzplace/results /home/jingqzha/projects/sglang/.jzplace/results_qps_new
