#!/usr/bin/env python3

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests

# Get current environment and ensure PYTHONPATH is preserved
ENV = os.environ.copy()

# List of experiments to run
EXPERIMENTS = {
    "None": "SGLang - Baseline",
    "EG,1_1_1": "SGLang with EAGLE,1_1_1",
    "EG,1_2_2": "SGLang with EAGLE,1_2_2",
    "EG,3_2_4": "SGLang with EAGLE,3_2_4",
    "EG,3_4_8": "SGLang with EAGLE,3_4_8",
    "EG,5_8_16": "SGLang with EAGLE,5_8_16",
    "EG,1_2_2,3_4_8,5_8_16": "SGLang with EAGLE & MAB",
}


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
        subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)  # Brief wait for port cleanup
    except:
        pass


def start_server(mab_config, args):
    """Start the server with the given configuration."""
    port = args.port
    model = args.model
    speculative_draft = args.speculative_draft
    tp_size = args.tp_size
    mem_fraction = args.mem_fraction

    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        f"--port={port}",
        f"--model={model}",
        f"--mem-fraction={mem_fraction}",
        "--random-seed=42",
        "--dtype=bfloat16",
        f"--tp-size={tp_size}",
    ]

    if mab_config.startswith("None"):
        options = mab_config.split(",")[1:]
        for option in options:
            cmd.extend([f"--{option}"])

    else:
        cmd.extend(
            [
                "--speculative-algo=EAGLE",
                f"--speculative-draft={speculative_draft}",
            ]
        )
        mab_algorithm, mab_configs = mab_config.split(",", 1)
        num_mab_strategies = len(mab_configs.split(","))
        if num_mab_strategies == 1:
            config_parts = mab_configs.split("_")
            assert len(config_parts) == 3, "Invalid config format"
            cmd.extend(
                [
                    f"--speculative-num-steps={config_parts[0]}",
                    f"--speculative-eagle-topk={config_parts[1]}",
                    f"--speculative-num-draft-tokens={config_parts[2]}",
                ]
            )
        elif num_mab_strategies > 1:
            cmd.extend(
                [
                    f"--speculative-eagle-mab-algorithm={mab_algorithm}",
                    f"--speculative-eagle-mab-configs={mab_configs}",
                ]
            )
            # Override default --speculative-num-steps, topk, and draft_tokens
            last_config = mab_configs.split(",")[-1]
            config_parts = last_config.split("_")
            cmd.extend(
                [
                    f"--speculative-num-steps={config_parts[0]}",
                    f"--speculative-eagle-topk={config_parts[1]}",
                    f"--speculative-num-draft-tokens={config_parts[2]}",
                ]
            )
    # Start process in its own process group
    cmd = [x for x in cmd if x != ""]
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


def run_benchmark(output_file, num_prompts, args):
    """Run the benchmark with specified parameters."""

    cmd_base = [
        "python3",
        "-m",
        "sglang.bench_serving",
        "--backend=sglang-oai",
        f"--dataset-name={args.dataset_name}",
        (
            f"--sharegpt-context-len={args.sharegpt_context_len}"
            if args.sharegpt_context_len is not None
            else ""
        ),
        (
            f"--sharegpt-output-len={args.sharegpt_output_len}"
            if args.sharegpt_output_len is not None
            else ""
        ),
        f"--port={args.port}",
        f"--output-file={output_file}",
        "--apply-chat-template",
        (
            f'--extra-request-body={{\"temperature\": {args.temperature}}}'
            if args.temperature is not None
            else ""
        ),
    ]

    # Create a temporary file with the modified get_request function for Poisson distribution
    if args.traffic_rate_option == "poisson":
        poisson_script_path = Path(args.results_dir) / "poisson_request_patch.py"
        with open(poisson_script_path, "w") as f:
            f.write("""
# Patch for get_request function to use Poisson distribution
import asyncio
import numpy as np
from typing import AsyncGenerator, List, Tuple

async def poisson_get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    cv: float = 1.0,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    \"\"\"Generate requests with Poisson arrival times.
    
    Args:
        input_requests: List of (prompt, prompt_len, output_len) tuples
        request_rate: Average number of requests per second
        cv: Coefficient of variation for controlling variability
    \"\"\"
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Use gamma distribution to simulate Poisson arrivals with controlled variability
        shape = 1 / (cv * cv)  # Shape parameter for gamma distribution
        scale = cv * cv / request_rate  # Scale parameter for gamma distribution
        
        # Sample the request interval from the gamma distribution
        interval = np.random.gamma(shape, scale)
        
        # The next request will be sent after the interval
        await asyncio.sleep(interval)

# Monkey patch the get_request function in bench_serving
import sys
import sglang.bench_serving
sglang.bench_serving.get_request = poisson_get_request
""")
        
        # Add the patch to PYTHONPATH
        ENV["PYTHONPATH"] = str(poisson_script_path.parent) + ":" + ENV.get("PYTHONPATH", "")
        
        # Add the -m option to run the patch before bench_serving
        cmd_base.insert(1, "-c")
        cmd_base.insert(2, f"import sys; sys.path.insert(0, '{poisson_script_path.parent}'); import poisson_request_patch; from sglang.bench_serving import main; main()")
        cmd_base.remove("sglang.bench_serving")
        
        # Add coefficient of variation parameter
        cv = args.poisson_cv if hasattr(args, 'poisson_cv') and args.poisson_cv is not None else 1.0
        ENV["POISSON_CV"] = str(cv)

    if args.traffic_rate_option == "qps":
        for request_rate in args.traffic_rate_list:
            cmd = cmd_base + [
                f"--num-prompts={max(100, int(num_prompts * min(1, request_rate/10)))}",
                "--max-concurrency=256",
                f"--request-rate={request_rate}",
            ]
            cmd = [x for x in cmd if x != ""]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True, env=ENV)
            time.sleep(10)
    elif args.traffic_rate_option == "concurrency":
        for concurrency in args.traffic_rate_list:
            cmd = cmd_base + [
                f"--num-prompts={max(100, int(num_prompts * min(1, concurrency/10)))}",
                f"--max-concurrency={concurrency}",
                "--request-rate=256",
            ]
            cmd = [x for x in cmd if x != ""]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True, env=ENV)
            time.sleep(10)
    elif args.traffic_rate_option == "poisson":
        for request_rate in args.traffic_rate_list:
            cmd = cmd_base + [
                f"--num-prompts={max(100, int(num_prompts * min(1, request_rate/10)))}",
                "--max-concurrency=256",
                f"--request-rate={request_rate}",
            ]
            cmd = [x for x in cmd if x != ""]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True, env=ENV)
            time.sleep(10)
    else:
        assert args.traffic_rate_option in ["qps", "concurrency", "poisson"], "Invalid options"


def plot_results(EXPERIMENTS, results_dir):
    """Generate plots from the results."""
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Set the figure background to white
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("white")

    # Color scheme from memory
    colors = [plt.cm.tab10(i % 10) for i in range(len(EXPERIMENTS))]

    for exp_idx, (exp, exp_name) in enumerate(EXPERIMENTS.items()):
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
            output_throughput = [d["output_throughput"] for d in data]
            input_throughput = [d["input_throughput"] for d in data]
            mean_otps = [1000 / d["mean_tpot_ms"] for d in data]
            mean_ttft_ms = [d["mean_ttft_ms"] for d in data]
            total_throughput = [
                i + o for i, o in zip(input_throughput, output_throughput)
            ]

            color = colors[exp_idx]

            # Plot the four subplots
            axes[0, 0].plot(
                output_throughput, mean_otps, "o-", label=exp_name, color=color
            )
            axes[0, 1].plot(
                output_throughput, mean_ttft_ms, "o-", label=exp_name, color=color
            )
            axes[1, 0].plot(
                total_throughput, mean_otps, "o-", label=exp_name, color=color
            )
            axes[1, 1].plot(
                total_throughput, mean_ttft_ms, "o-", label=exp_name, color=color
            )

        except Exception as e:
            print(f"Error processing {exp}: {str(e)}")
            continue

    # Configure the plots
    for ax in axes.flat:
        ax.grid(True)
        ax.legend()
        # ax.set_xscale('log')  # Use log scale for time metrics
        # ax.set_yscale('log')  # Use log scale for throughput

    axes[0, 0].set_ylabel("Mean Inference Speed")
    axes[0, 0].set_xlabel("Output Throughput")
    axes[0, 1].set_ylabel("Mean TTFT (ms)")
    axes[0, 1].set_xlabel("Output Throughput")
    axes[0, 1].set_ylim([1, 100])

    axes[1, 0].set_ylabel("Mean Inference Speed")
    axes[1, 0].set_xlabel("Total Throughput")
    axes[1, 1].set_ylabel("Mean TTFT (ms)")
    axes[1, 1].set_xlabel("Total Throughput")
    axes[1, 1].set_ylim([1, 100])

    plt.tight_layout()
    plt.savefig(
        Path(results_dir) / "speculative_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_strategy_selection(exp, results_dir, args):
    try:
        from sglang.srt.speculative.eagle_mab import MABGroupManager
        
        # Try to access the MAB instance via HTTP
        url = f"http://{args.host}:{args.port}/get_mab_stats"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"MAB stats request successful: {response.json()}")
            else:
                print(f"Failed to get MAB stats: {response.status_code}")
        except Exception as e:
            print(f"Error requesting MAB stats: {e}")
            
        # As a fallback, try to directly trigger the plotting via the server process
        # This is a more direct approach that doesn't rely on HTTP
        try:
            # Create a socket connection to the server
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Connect to the server on a special port (main port + 1000)
                s.connect((args.host, args.port + 1000))
                # Send the command to get MAB stats
                s.sendall(b"GET_MAB_STATS\n")
                # Receive the response
                response = s.recv(1024)
                print(f"Socket response: {response.decode('utf-8')}")
        except Exception as e:
            print(f"Socket communication failed: {e}")
            
            # As a last resort, create a direct plot file
            plot_path = Path(results_dir) / f"{exp}_strategy_selection.png"
            print(f"Creating placeholder plot at {plot_path}")
            
            # Create a simple placeholder plot
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "MAB Strategy Selection\n(Placeholder - Server Communication Failed)", 
                    horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.axis('off')
            plt.savefig(plot_path)
            plt.close()
            
    except ImportError as e:
        print(f"Failed to import MAB module: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--results-dir", type=str, default="mab_results")
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument(
        "--traffic-rate-option",
        type=str,
        default="concurrency",
        choices=["qps", "concurrency", "poisson"],
    )
    parser.add_argument("--traffic-rate-list", type=str, default=None)
    parser.add_argument(
        "--dataset-name", type=str, default="sharegpt", choices=["sharegpt"]
    )
    parser.add_argument(
        "--sharegpt-context-len",
        type=int,
        default=None,
        help="Requests longer than the context length will be dropped.",
    )
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Requests shorter than this length will be dropped.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Overrides the output length from the ShareGPT dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="The LLM model to use.",
    )
    parser.add_argument(
        "--mem-fraction",
        type=float,
        default=0.7,
        help="The CUDA memory fraction to use.",
    )
    parser.add_argument(
        "--speculative-draft",
        type=str,
        default="lmzheng/sglang-EAGLE-llama2-chat-7B",
        help="The speculative draft model to use.",
    )
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor Parallel Size")
    parser.add_argument(
        "--poisson-cv",
        type=float,
        default=None,
        help="Coefficient of variation for Poisson distribution",
    )
    parser.add_argument("--host", type=str, default="localhost")

    args = parser.parse_args()
    print(args)

    if args.traffic_rate_list is not None:
        args.traffic_rate_list = [int(x) for x in args.traffic_rate_list.split(",")]
    elif args.traffic_rate_option == "qps":
        args.traffic_rate_list = [0.1, 1, 4, 16, 64, 256]
    else:
        args.traffic_rate_list = [1, 4, 16, 64, 256]

    folder = args.traffic_rate_option + "_" + args.dataset_name
    if args.sharegpt_context_len is not None:
        folder += f"_context{args.sharegpt_context_len}"
    if args.sharegpt_output_len is not None:
        folder += f"_output{args.sharegpt_output_len}"
    if args.temperature is not None:
        folder += f"_temperature{args.temperature}"
    args.results_dir = args.results_dir + "/" + folder
    ENV["MAB_RESULTS_DIR"] = args.results_dir

    # Create results directory if it doesn't exist
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {args.results_dir}")
    original_mem_fraction = args.mem_fraction
    for exp in EXPERIMENTS.keys():
        print(f"\nRunning experiment: {exp}")

        # Kill any existing processes on port
        kill_server(port=args.port)

        # Run more warmup in case MAB as it requires a warm start
        if not exp.startswith("None") and len(exp.split(",")) > 2:
            warmup_iter = 2
            args.mem_fraction = original_mem_fraction - 0.1
        else:
            warmup_iter = 1
            args.mem_fraction = original_mem_fraction

        try:
            server_process = None
            for i in range(warmup_iter + 1):
                if i < warmup_iter:
                    result_file = Path(args.results_dir) / f"{exp}_warmup_{i+1}.jsonl"
                else:
                    result_file = Path(args.results_dir) / f"{exp}.jsonl"

                if result_file.exists():
                    continue

                if server_process is None:
                    server_process = start_server(exp, args)

                run_benchmark(
                    output_file=str(result_file),
                    num_prompts=args.num_prompts,
                    args=args,
                )
                
                # Plot MAB strategy selection if using MAB
                if not exp.startswith("None") and len(exp.split(",")) > 2:
                    plot_strategy_selection(exp, args.results_dir, args)

        except Exception as e:
            print(f"Error running experiment: {str(e)}")
            # remove result_file as it is not complete
            if result_file.exists():
                result_file.unlink()
            kill_server(process=server_process, port=args.port)
        finally:
            # Always kill the server when done
            kill_server(process=server_process, port=args.port)

        # Generate plots
        print("\nGenerating plots...")
        plot_results(EXPERIMENTS, args.results_dir)

    print("Done! Results are in the results directory.")


if __name__ == "__main__":
    main()
