# Install Dependencies
Follow the general installation guide at https://docs.sglang.ai/start/install.html#method-2-from-source.

See special notes below.
```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Checkout the commit after EAGLE2 refactoring (Feb 3)
# Later commits introduced bugs (illegal CUDA memory access) when request concurrency is high.
# Reference: https://github.com/sgl-project/sglang/commits/main/python/sglang/srt/speculative/eagle_worker.py
git checkout 013021b 

conda create -n sglang-eagle python=3.12 -y
conda activate sglang-eagle

# Install required packages
pip install --upgrade pip
pip install sgl-kernel --force-reinstall --no-deps
pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/

# Required by EAGLE in this sglang version. It might lead to issue on H100 (`Value 'sm_90' is not defined for option 'gpu-architecture'`). See the section of "Issue" for a fix below.
pip install cutex

# Install matplotlib for debugging and data analysis
pip install matplotlib
```

# Test EAGLE

## Step 1: Start the server

```bash
# With speculative decoding and port 30023
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model meta-llama/Llama-2-7b-chat-hf \
    --speculative-algo EAGLE \
    --speculative-draft lmzheng/sglang-EAGLE-llama2-chat-7B \
    --speculative-num-steps=3 \
    --speculative-eagle-topk=4 \
    --speculative-num-draft-tokens=8 \
    --mem-fraction=0.7 \
    --port=30022

# Without speculative decoding and with port 30023
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server --model meta-llama/Llama-2-7b-chat-hf \
    --mem-fraction=0.7 \
    --port=30023
```

## Step 2: Start Benchmarking

Speculative decoding performs better when the draft model and target model make consistent predictions. This typically occurs when:
- Both models apply the chat template (e.g., [INST] and [/INST] for Llama3.1-8B models)
- Requests use lower temperature settings (less creative outputs)
- Requests focus on coding/fact checking rather than creative tasks

To validate these concepts:
### Test simple prompts
```bash
python .jzplace/eagle_testing_simple.py
```

### Test with sharegpt dataset
```bash
# Set backend=sglang-oai to apply the chat template, so that draft and target can produce good and consistent results, because draft model is trained this way
python3 -m sglang.bench_serving \
    --backend sglang-oai \
    --dataset-name sharegpt \
    --num-prompts 100 --max-concurrency 256  \
    --request-rate-range 1,4,16,64,256 --multi \
    --port 30022 \
    --output-file .jzplace/benchmark_w_speculative_3_4_8_oai.jsonl

# With backend=sglang, draft and target models have less change to have consistent predictions
python3 -m sglang.bench_serving \
    --backend sglang-oai \
    --dataset-name sharegpt \
    --num-prompts 100 --max-concurrency 256  \
    --request-rate-range 1,4,16,64,256 --multi \
    --port 30023 \
    --output-file .jzplace/benchmark_wo_speculative.jsonl
```

# Test EAGLE + MAB
## Step 1: Start the server
```
python -m sglang.launch_server --model meta-llama/Llama-2-7b-chat-hf \
    --speculative-algo EAGLE \
    --speculative-draft lmzheng/sglang-EAGLE-llama2-chat-7B \
    --speculative-num-steps=3 --speculative-eagle-topk=4 --speculative-num-draft-tokens=8 \
    --speculative-eagle-mab=EG,1_2_2,3_2_4,3_4_8,5_8_16 \
    --mem-fraction=0.7 \
    --port=30023
```
## Step 2: Start Benchmarking
Same process as described above.

# Test and compare different MAB settings
The following script compares various MAB settings:

```bash
# To test different concurrencies
python .jzplace/eagle_testing.py --traffic-rate-option=concurrency

# To test different qps
python .jzplace/eagle_testing.py --traffic-rate-option=qps
```

The experiments test different configurations:
- `EG`: Epsilon Greedy algorithm
- `UCB1`: Upper Confidence Bound algorithm
- `None`: No speculative decoding
- Notation format (e.g., `3_4_8`): represents `speculative-num-steps=3 --speculative-eagle-topk=4 --speculative-num-draft-tokens=8`

```python
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
```

# Issues
## CUDA Architecture Fix for H100 GPUs
Note: This fix is no longer required in newer sglang versions.

If you're using H100 GPUs and encounter CUDA architecture errors (`Value 'sm_90' is not defined for option 'gpu-architecture'`), you need to modify the CUDA configuration in the cutex module. Edit the file `site-packages/cutex/src_module.py` in your conda environment:

```python
# In _jit_compile method of SourceModule class:

# Set NVCC path to CUDA 12.6 (or your CUDA version that supports H100)
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.6'
os.environ['PATH'] = '/usr/local/cuda-12.6/bin:' + os.environ['PATH']

# Set architecture for H100
arch_flag = 'sm_90'

# Update compile options with proper architecture
if 'arch' not in self.compile_kwds:
    self.compile_kwds['arch'] = arch_flag
```