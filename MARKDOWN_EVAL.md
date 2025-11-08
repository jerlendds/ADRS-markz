# md_wasm Evolution Spaces (1 by a human, 3 created by AI)

## Evolution Space 0 _ChatGPT5 Generated from the prompt below_

```bash
python openevolve-run.py \
  --initial_program_path openevolve/examples/ADRS/md_wasm_0/initial_program.py \
  --evaluation_file openevolve/examples/ADRS/md_wasm_0/evaluator.py \
  --config_path openevolve/examples/ADRS/md_wasm_0/config.yaml
```

## Evolution Space 1 _ChatGPT5 Generated from the prompt below_

```bash
python openevolve-run.py \
  --initial_program_path openevolve/examples/ADRS/md_wasm_1/initial_program.py \
  --evaluation_file openevolve/examples/ADRS/md_wasm_1/evaluator.py \
  --config_path openevolve/examples/ADRS/md_wasm_1/config.yaml
```

## Evolution Space 2 _ChatGPT5 Generated from the prompt below_

```bash
python openevolve-run.py \
  --initial_program_path openevolve/examples/ADRS/md_wasm_2/initial_program.py \
  --evaluation_file openevolve/examples/ADRS/md_wasm_2/evaluator.py \
  --config_path openevolve/examples/ADRS/md_wasm_2/config.yaml
```

## Evolution Space 3 _Human (jerlendds) wrote this program_

TODO... _;)_

---

# Prompt used to generate 3 evolution spaces:

Task: Create the `initial_program_path`, `evaluation_file` and `config_path` for an AlphaEvolve-like system named OpenEvolve. The goal is to evolve an extremely fast Markdown parser & HTML renderer implemented in WebAssembly. The program requirements are to use Rust and be compliant with the CommonMark specification for markdown.
Evidence:

- Relevant files from examples in https://github.com/UCB-ADRS/ADRS ```
  ├── openevolve
  │ ├── examples
  │ │ ├── ADRS
  │ │ │ ├── hp_quantization
  │ │ │ │ ├── bq_evaluator.py
  │ │ │ │ ├── bq_openevolve_config.yaml
  │ │ │ │ ├── bq_program.py
  │ │ │ │ ├── README.md
  │ │ │ │ └── utils.py
  │ │ │ ├── llm_sql
  │ │ │ │ ├── datasets
  │ │ │ │ │ ├── beer.csv
  │ │ │ │ │ ├── BIRD.csv
  │ │ │ │ │ ├── movies.csv
  │ │ │ │ │ ├── PDMX.csv
  │ │ │ │ │ └── products.csv
  │ │ │ │ ├── config.yaml
  │ │ │ │ ├── evaluator.py
  │ │ │ │ ├── initial_program.py
  │ │ │ │ ├── quick_greedy.py
  │ │ │ │ ├── solver.py
  │ │ │ │ └── utils.py
  │ ├── cli.py
  │ ├── config.py
  │ ├── controller.py
  │ ├── database.py
  │ ├── evaluation_result.py
  │ ├── evaluator.py
  │ ├── iteration.py
  │ └── README.md
  ├── openevolve-run.py
  ├── pyproject.toml
  ├── README.md
  ├── setup.py
  └── uv.lock

````

# File Contents

## openevolve/examples/ADRS/hp_quantization/bq_evaluator.py

```python
"""
Evaluator for circle packing example (n=26) with improved timeout handling
"""

import importlib.util
import numpy as np
import time
import os
import signal
import subprocess
import tempfile
import traceback
import sys
import pickle


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def run_with_timeout(program_path, timeout_seconds=1200):
    """
    Run the program in a separate process with timeout
    using a simple subprocess approach

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds

    Returns:
        centers, radii, sum_radii tuple from the program
    """
    # Create a temporary file to execute
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        # Write a script that executes the program and saves results
        script = f"""
import sys
import numpy as np
import os
import pickle
import traceback

# Add the directory to sys.path
sys.path.insert(0, os.path.dirname('{program_path}'))

# Debugging info
print(f"\033[92mRunning {program_path} in subprocess, Python version: {sys.version}\033[0m")

try:
    # Import the program
    spec = __import__('importlib.util').util.spec_from_file_location("program", '{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)

    # Run the compression function
    print("\033[92mCalling compression()...\033[0m")
    output_dict = program.run_compression()
    print(f"\033[92mcompression() returned successfully: bitrate = {{output_dict['bitrate']}}\033[0m")

    # Save results to a file
    results = output_dict

    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {temp_file.name}.results")

except Exception as e:
    # If an error occurs, save the error instead
    print(f"Error in subprocess: {{str(e)}}")
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
    print(f"Error saved to {temp_file.name}.results")
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        # Run the script with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            # Always print output for debugging purposes
            print(f"Subprocess stdout: {stdout.decode()}")
            if stderr:
                print(f"Subprocess stderr: {stderr.decode()}")

            # Still raise an error for non-zero exit codes, but only after printing the output
            if exit_code != 0:
                raise RuntimeError(f"Process exited with code {exit_code}")

            # Load the results
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

                # Check if an error was returned
                if "error" in results:
                    raise RuntimeError(f"Program execution failed: {results['error']}")

                return results
            else:
                raise RuntimeError("Results file not found")

        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)

def get_combined_score(bitrate, ptb_ppl, wikitext_ppl):
    """
    Get the combined score for the program
    Prioritizes: lower wikitext ppl > lower ptb ppl > lower bitrate
    Returns 0 if bitrate > 2.6 or wikitext ppl > 40
    """
    # Hard cutoffs
    if bitrate > 2.6 or wikitext_ppl > 10.5 or ptb_ppl > 18:
        return 0.0

    # Normalize metrics to 0-1 range (lower is better)
    # Assuming reasonable ranges: wikitext_ppl: 6-10.5, ptb_ppl: 10-18, bitrate: 2.0-2.6
    normalized_wikitext = max(0, 1 - (wikitext_ppl - 6) / 4.5)  # 6=1.0, 10.5=0.0
    normalized_ptb = max(0, 1 - (ptb_ppl - 10) / 8)  # 15=1.0, 18=0.0
    normalized_bitrate = max(0, 1 - (bitrate - 2.0) / 0.6)  # 2.0=1.0, 2.6=0.0

    # Weighted combination with priorities
    # wikitext ppl has highest priority (weight 0.5)
    # ptb ppl has medium priority (weight 0.1)
    # bitrate has lowest priority (weight 0.4)
    combined_score = (0.5 * normalized_wikitext +
                     0.3 * normalized_ptb +
                     0.2 * normalized_bitrate)

    return combined_score

def evaluate(program_path):
    """
    Evaluate the program by running it once and checking the sum of radii

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """

    try:
        # For constructor-based approaches, a single evaluation is sufficient
        # since the result is deterministic
        start_time = time.time()

        # Use subprocess to run with timeout
        output_dict = run_with_timeout(
            program_path, timeout_seconds=300  # Single timeout
        )

        end_time = time.time()
        eval_time = end_time - start_time

        bitrate = float(output_dict['bitrate'])
        wikitext_ppl = float(output_dict['wikitext_ppl'])
        ptb_ppl = float(output_dict['ptb_ppl'])

        # Combined score - higher is better
        combined_score = get_combined_score(bitrate, ptb_ppl, wikitext_ppl)

        print(
            f"\033[94mEvaluation: bitrate={bitrate:.6f}, ptb_ppl={ptb_ppl:.6f}, wikitext_ppl={wikitext_ppl:.6f}, combined_score={combined_score:.6f}, time={eval_time:.2f}s\033[0m"
        )

        return {
            "bitrate": float(bitrate),
            "ptb_ppl": float(ptb_ppl),
            "wikitext_ppl": float(wikitext_ppl),
            "eval_time": float(eval_time),
            "combined_score": float(combined_score),
        }

    except Exception as e:
        print(f"\033[91mEvaluation failed completely: {str(e)}\033[0m")
        traceback.print_exc()
        return {
            "bitrate": 0.0,
            "ptb_ppl": 0.0,
            "wikitext_ppl": 0.0,
            "eval_time": 0.0,
            "combined_score": 0.0,
        }

````

## openevolve/examples/ADRS/hp_quantization/bq_openevolve_config.yaml

```yaml
max_iterations: 500 # Increased iterations
checkpoint_interval: 5
log_level: "INFO"

# LLM configuration
llm:
  primary_model: "gemini-2.5-pro"
  primary_model_weight: 1.0
  secondary_model: "gemini-2.5-flash"
  secondary_model_weight: 0.2
  temperature: 0.7
  top_p: 0.95
  max_tokens: 32768
  timeout: 600
  api_base: "https://generativelanguage.googleapis.com/v1beta/openai/" # Base URL for API (change for non-OpenAI models)
  api_key: ${GEMINI_API_KEY}

# Prompt configuration
prompt:
  system_message:
    You are an expert search unit specializing in data compression methods using adaptive bitrate methods.
    Your task is to find the best bitrate configuration for the given LLM layer, based on the input and weight
    tensor. There are two main tuning knobs - high_bitrate_ratio and alpha. high_bitrate_ratio is used to determine the number
    of columns that are assigned higher bitrate (4 bits per element) compared to lower bitrate columns (2 bits per element),
    where as alpha balances and mixes the magnitude score and the Hessian score of each column when determining the importance of each column.
    The columns are sorted based on the importance score, and the first first high_bitrate_ratio * num_columns columns are assigned higher bitrate.

    Your goal is to implement an algorithm inside adaptive_bitrate_model() function optimizes the high bitrate column selection and ratio given an input and weights,
    and ultimately achieves low bitrate and ppl values for the model.

    Start by sweeping over different alpha values on each kurtosis range, and find the sweet spot of alpha on each range.
    Then, slowly adjust the bitrate by shaving off or adding small amounts of high_bitrate_ratio to increase accuracy. One way to achieve this is finding a
    pair of kurtosis ranges where shaving off large amount of bitrate from one range and adding back a small amount on another maintains the accuracy due to different importance
    of the bitrate and the associated bitrate sweetspot of the range. You can also experiment with using different configurations
    depending on the index of the current layer, provided using the layer_idx argument to the function.

    The function must return the high_bitrate_ratio float value between 0.0 and 1.0, and the hybrid_rank tensor which has
    the column indexes sorted based on their importance score. As the function is exported to a separate file to be used in
    other processes, it must be formatted as a python string.

    The evaluator will return a combined score of 0 if the bitrate is over 2.6 or the wikitext_ppl is over 10.5 or the ptb_ppl is over 18.
    The evaluator has a weight of 0.5 for wikitext_ppl score, 0.2 for bitrate, and 0.3 for ptb_ppl score.

  num_top_programs: 3
  num_diverse_programs: 2

# Database configuration
database:
  db_path: "./openevolve_output/bq_evolution"
  population_size: 80
  archive_size: 30
  num_islands: 5
  elite_selection_ratio: 0.3
  exploitation_ratio: 0.65
  exploration_ratio: 0.35

# Evaluator configuration
evaluator:
  timeout: 900
  parallel_evaluations: 1
  use_llm_feedback: false

# Evolution settings
max_code_length: 30000
diff_based_evolution: true # Use full rewrites instead of diffs
allow_full_rewrites: false # Allow full rewrites for constructor functions
```

## openevolve/examples/ADRS/hp_quantization/bq_program.py

```python
import os
import time
from utils import g_str, r_str, y_str, b_str
import subprocess
import torch
import threading
import numpy as np

# EVOLVE-BLOCK-START
# The code is written to a file and then imported in ptq.py.
# Therefore, the code must be written in a string.
adaptive_bitrate_model_code = """
import torch

def adaptive_bitrate_model(inputs, weights, hessian_score, magnitude_score, layer_idx):
    \"\"\"
    This function is called for each layer in an LLM to adaptively quantize its weight matrix.
    The input to the function is the calibration input tensor and the weight matrix of the layer.
    Using the data from the calibration input and the weight matrix, it finds the optimal high-bitrate ratio and the columns to assign higher bitrate when quantized.'
    The quantization algorithm takes the num_columns * high_bitrate_ratio columns with the highest hybrid score and assigns them higher bitrate.
    \"\"\"

    def get_tensor_info(tensor):
        \"\"\"Get distribution info of the tensor.\"\"\"
        tensor_flat = tensor.flatten().to(torch.float64)
        if tensor_flat.shape[0] > 2**24:
            sample_idx = torch.randint(0, tensor_flat.shape[0], (2**24,))
            tensor_flat = tensor_flat[sample_idx]
        num = tensor_flat.size(0)
        avg = torch.mean(tensor_flat)
        min_val = torch.min(tensor_flat)
        max_val = torch.max(tensor_flat)
        std = torch.std(tensor_flat)
        skew = torch.mean((tensor_flat - avg) ** 3) / (std ** 3)
        kurtosis = torch.mean((tensor_flat - avg) ** 4) / (std ** 4)
        per_001 = torch.quantile(tensor_flat, 0.001)
        per_01 = torch.quantile(tensor_flat, 0.01)
        per_25 = torch.quantile(tensor_flat, 0.25)
        per_50 = torch.quantile(tensor_flat, 0.50)
        per_75 = torch.quantile(tensor_flat, 0.75)
        per_99 = torch.quantile(tensor_flat, 0.99)
        per_999 = torch.quantile(tensor_flat, 0.999)
        return {"num": num, avg: avg,
            "min": min_val, "max": max_val,
            "std": std.item(), "skew": skew.item(),
            "kurtosis": kurtosis, "per_001": per_001,
            "per_01": per_01, "per_25": per_25,
            "per_50": per_50, "per_75": per_75,
            "per_99": per_99, "per_999": per_999}
    w_info = get_tensor_info(weights)
    w_kurtosis = w_info['kurtosis']
    i_info = get_tensor_info(inputs)
    i_kurtosis = i_info['kurtosis']
    if torch.is_tensor(w_kurtosis):
        w_kurtosis = w_kurtosis.item()
    if torch.is_tensor(i_kurtosis):
        i_kurtosis = i_kurtosis.item()

    if w_kurtosis > 12 or i_kurtosis > 12:
        high_bitrate_ratio = 0.08
        alpha = 0.5
    elif w_kurtosis > 8 or i_kurtosis > 8:
        high_bitrate_ratio = 0.05
        alpha = 0.7
    elif w_kurtosis > 6 or i_kurtosis > 6:
        high_bitrate_ratio = 0.08
        alpha = 0.7
    elif w_kurtosis > 5:
        high_bitrate_ratio = 0.10
        alpha = 0.75
    elif w_kurtosis > 4:
        high_bitrate_ratio = 0.25
        alpha = 0.75
    elif w_kurtosis > 3.5:
        high_bitrate_ratio = 0.35
        alpha = 0.8
    elif w_kurtosis > 3.05:
        high_bitrate_ratio = 0.52
        alpha = 0.9
    else:
        high_bitrate_ratio = 0.12
        alpha = 1.0


    hybrid_score = alpha * hessian_score + (1 - alpha) * magnitude_score
    hybrid_rank = torch.argsort(hybrid_score, descending=True)

    return high_bitrate_ratio, hybrid_rank
"""
# EVOLVE-BLOCK-END

def run_compression():
    model_name = "meta-llama/Meta-Llama-3-8B"
    rotation = "8B_R.bin"
    # model_name = "meta-llama/Llama-3.2-1B"
    # rotation = "1B_R.bin"
    method = "bq"
    key_bits = 16
    V = 8
    do_gptq = True

    adaptive_bitrate_model_path = "adaptive_bitrate_model.py"
    with open(adaptive_bitrate_model_path, "w") as f:
        f.write(adaptive_bitrate_model_code)

    def print_log(message):
        print(message) # Print to console with colors

    def stream_reader_thread(stream, stream_name_for_log, output_list,
                            print_log_func, color_func=None):
        try:
            for line in iter(stream.readline, ''): # Read until pipe closes
                if not line: # End of stream
                    break
                line = line.rstrip()
                log_prefix = f"[{stream_name_for_log}] "
                if color_func:
                    print_log_func(color_func(log_prefix) + line)
                else:
                    # For stdout, child's output might already be colored
                    print_log_func(line)
                output_list.append(line)
        except ValueError:
            # Can happen if pipe is closed abruptly
            print_log_func(r_str(f"[{stream_name_for_log}] Pipe closed or "
                                f"encoding error."))
        except Exception as e:
            print_log_func(
                r_str(f"[{stream_name_for_log}] Error reading stream: {e}")
            )

    iter_start_time = time.time()
    current_time_str = time.strftime("%H-%M-%S", time.localtime())
    num_gpus = torch.cuda.device_count()

    log_msg_header = b_str(
        f"[{current_time_str}] Running: "
    )
    log_msg_details = y_str(
        f"{model_name}, {method}, k={key_bits}, V={V}"
    )
    print_log(log_msg_header + log_msg_details)

    current_port = 25000 + (os.getpid() % 1000) * 20

    cmd = \
        f"torchrun --nnodes=1 --nproc_per_node={num_gpus} " + \
        f"--master_port={current_port} " + \
        f"ptq.py --input_model {model_name} " + \
        f"--do_train False --do_eval True " + \
        f"--per_device_eval_batch_size 4 " + \
        f"--model_max_length 2048 " + \
        f"--save_safetensors False " + \
        f"--w_bits {key_bits} " + \
        f"--w_clip " + \
        f"--w_groupsize {V} " + \
        f"--rotate " + \
        f"--save_qmodel_path {model_name}_{method}_{key_bits}_{V}_{do_gptq} " + \
        f"--optimized_rotation_path {rotation} "
    if not do_gptq:
        cmd += "--no_gptq "
    if method != "dummy":
        cmd += f"--use_{method} "

    print_log(g_str(f"Command:"))
    print_log(cmd)

    current_process = None
    # Initialize results for this iteration
    bitrate = "N/A"
    wikitext_ppl, ptb_ppl, c4_ppl = "N/A", "N/A", "N/A"
    c_qa, arc_c, arc_e = "N/A", "N/A", "N/A"
    hs, piqa, winogrande, avg_zs = "N/A", "N/A", "N/A", "N/A"
    time_taken_str = "N/A"
    status_for_csv = "STARTED"

    stdout_lines_list = []
    stderr_lines_list = []
    stdout_thread = None
    stderr_thread = None

    try:
        child_pid_str = (
            f"(PID: {current_process.pid if current_process else 'N/A'})"
        )
        print_log(g_str(f"--- Child Process Output {child_pid_str} ---"))

        current_process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid if os.name != 'nt' else None
        )
        # Update PID string after Popen
        child_pid_str = f"(PID: {current_process.pid})"
        print_log(g_str(f"--- Child Process Output {child_pid_str} ---"))


        if current_process.stdout:
            stdout_thread = threading.Thread(
                target=stream_reader_thread,
                args=(current_process.stdout, "stdout",
                    stdout_lines_list, print_log)
            )
            stdout_thread.start()

        if current_process.stderr:
            stderr_thread = threading.Thread(
                target=stream_reader_thread,
                args=(current_process.stderr, "stderr",
                    stderr_lines_list, print_log, r_str)
            )
            stderr_thread.start()

        if stdout_thread: stdout_thread.join()
        if stderr_thread: stderr_thread.join()

        current_process.wait() # Wait for process to terminate
        return_code = current_process.returncode
        log_msg = (f"--- Child Process {child_pid_str} Finished "
                f"(Return Code: {return_code}) ---")
        print_log(g_str(log_msg))

        if return_code != 0:
            err_msg = (f"MAIN: Child for {model_name} exited with "
                    f"error code {return_code}.")
            print_log(r_str(err_msg))

        for line in stdout_lines_list:
            if "Average bitrate: " in line:
                bitrate = line.split("Average bitrate: ")[-1].strip()
            if "wikitext: " in line:
                wikitext_ppl = line.split("wikitext: ")[-1].strip()
            if "ptb: " in line:
                ptb_ppl = line.split("ptb: ")[-1].strip()

    except Exception as e:
        err_msg = (f"Error during subprocess execution or parsing "
                f"for {model_name}:")
        print_log(r_str(err_msg))
        print_log(r_str(f"Exception type: {type(e).__name__}, Msg: {e}"))
        import traceback
        detailed_error_info = traceback.format_exc()
        print_log(r_str("MAIN SCRIPT TRACEBACK FOR ITERATION ERROR:"))
        print_log(detailed_error_info)
        # Cleanup threads and process if they exist
        if stdout_thread and stdout_thread.is_alive():
            stdout_thread.join(timeout=2)
        if stderr_thread and stderr_thread.is_alive():
            stderr_thread.join(timeout=2)
        if current_process and current_process.poll() is None:
            current_process.kill()
            current_process.wait()

    finally: # Per-iteration finally block
        iter_end_time = time.time()
        time_taken_seconds = iter_end_time - iter_start_time
        time_taken_str = f"{time_taken_seconds:.2f}s"
        print_log(f"Iteration time: {time_taken_str}")

        output_dict = {
            "model_name": model_name,
            "method": method,
            "key_bits": key_bits,
            "V": V,
            "bitrate": bitrate,
            "do_gptq": do_gptq,
            "time_taken_str": time_taken_str,
            "wikitext_ppl": wikitext_ppl,
            "ptb_ppl": ptb_ppl,
        }

    return output_dict

if __name__ == "__main__":
    run_compression()
```

## openevolve/examples/ADRS/llm_sql/config.yaml

````yaml
# Configuration for LLM-Retrieval Optimization
# Objective: Optimize DataFrame column ordering to maximize prefix hit count for LLM prompt caching
# Input: DataFrame with rows and columns containing text data
# Output: Optimized DataFrame with reordered columns that maximize prefix reuse

max_iterations: 100
checkpoint_interval: 10
log_level: "INFO"

# LLM configuration
llm:
  models:
    - name: "gemini-2.5-pro"
      weight: 0.2 # Lower weight for the conservative model
      api_base: "https://generativelanguage.googleapis.com/v1beta/openai/"
      api_key: ${GEMINI_API_KEY}
    - name: "o3"
      weight: 0.8 # Higher weight for the creative/exploratory model
      api_base: "https://api.openai.com/v1"
      api_key: ${OPENAI_API_KEY}
  temperature: 0.7 # Reduced from 0.95 for more focused generation
  top_p: 0.95
  max_tokens: 32000
  timeout: 600

# Prompt configuration
prompt:
  system_message: |
    You are an expert in data optimization and LLM prompt caching. Your task is to evolve the existing Evolved class to maximize prefix hit count (PHC) for efficient LLM prompt caching.

    Problem Context:
    - You are given a pandas DataFrame `df` with text data in rows and columns
    - The goal is to reorder columns to maximize prefix reuse when processing rows sequentially
    - Prefix reuse occurs when consecutive rows have matching values in the same column positions
    - This reduces LLM computation costs by reusing cached prefixes

    Objective:
    - Dual objective: (1) maximize prefix reuse across consecutive rows and (2) minimize end-to-end runtime of the algorithm.
    - Your goal is to evolve the Evolved class such that when the LLM processes each row sequentially, it reuses as much of the prefix from the previous row as possible, while keeping the algorithm computationally efficient.
    - Prefix reuse is defined as consecutive field values (starting from the first column) that are **exact matches** with the corresponding fields of the previous row.
    - The **hit score** of a row is defined as the **sum of squares of the string lengths** of the matching prefix fields.
    - The algorithm will be evaluated on a combined metric that balances accuracy (prefix reuse) and speed (runtime).

    Formally:
    - For a given column ordering `C`, PHC(C) = sum over all rows `r` of `hit(C, r)`
    - `hit(C, r)` = sum of `len(df[r][C[f]])^2` for all f in prefix where `df[r][C[f]] == df[r-1][C[f]]`; zero if mismatch starts at the first field.
    - Runtime is measured as wall-clock seconds to compute the reordered DataFrame from the input DataFrame.
    - Combined score used for selection: `combined_score = 0.95 * average_hit_rate + 0.05 * (12 - min(12, average_runtime)) / 12`.

    Required API (DO NOT CHANGE):
    - You must keep the existing Evolved class structure and the reorder method signature:
      ```python
      class Evolved(Algorithm):
          def reorder(
              self,
              df: pd.DataFrame,
              early_stop: int = 0,
              row_stop: int = None,
              col_stop: int = None,
              col_merge: List[List[str]] = [],
              one_way_dep: List[Tuple[str, str]] = [],
              distinct_value_threshold: float = 0.8,
              parallel: bool = True,
          ) -> Tuple[pd.DataFrame, List[List[str]]]:
      ```
    - You can modify the internal implementation of methods but must preserve the class structure and method signatures
    - The reorder method must return a tuple of (reordered_dataframe, column_orderings)

    Algorithm Design Guidelines:
    - For each row, determine the optimal column order based on matches with the previous row
    - Consider column statistics (unique values, string lengths) for ordering
    - Implement greedy or heuristic approaches for scalability
    - Focus on columns with high value frequency and long strings
    - Handle missing values and mixed data types appropriately
    - Optimize the existing recursive approach or replace it with more efficient vectorized methods
    - Consider prefix-aware greedy approaches that condition on the current matched prefix

    Constraints:
    - Only reorder columns, do not change row order or add/remove rows or columns
    - You must have different column orderings for different rows to maximize prefit hit rate
    - Return a DataFrame with the same shape as input
    - Use exact string matching for prefix calculations
    - Keep memory usage reasonable for large datasets
    - Preserve all existing method signatures and class structure
    - The algorithm will be called with the same parameters as the original Evolved

    Simply return the optimized Evolved class, do not provide explanations.
  num_top_programs: 3
  use_template_stochasticity: true

# Database configuration
database:
  population_size: 50
  archive_size: 20
  num_islands: 3
  elite_selection_ratio: 0.2
  exploitation_ratio: 0.7

# Evaluator configuration
evaluator:
  timeout: 60
  cascade_evaluation: false
  cascade_thresholds: [0.5, 0.75]
  parallel_evaluations: 4
  use_llm_feedback: false

# Evolution settings
diff_based_evolution: true
allow_full_rewrites: false
max_code_length: 60000
````

## openevolve/examples/ADRS/llm_sql/evaluator.py

```python
import sys
import os
import traceback
import time

import pandas as pd
from pandas.io.sql import com

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
import importlib.util

from utils import evaluate_df_prefix_hit_cnt
from initial_program import Evolved
from quick_greedy import QuickGreedy


def run_quick(
    master_df,
    col_merge,
):
    st = time.time()
    quick, _ = QuickGreedy().reorder(
        master_df,
        early_stop=100000,
        distinct_value_threshold=0.7,
        row_stop=4,
        col_stop=2,
        col_merge=col_merge,
    )
    end = time.time() - st

    results = evaluate_df_prefix_hit_cnt(quick)
    return results, end

def run_evolved(
    master_df,
    col_merge,
):
    st = time.time()
    reordered, _ = Evolved().reorder(
        master_df,
        early_stop=100000,
        distinct_value_threshold=0.7,
        row_stop=4,
        col_stop=2,
        col_merge=col_merge,
    )
    end = time.time() - st

    results = evaluate_df_prefix_hit_cnt(reordered)
    return results, end


def run(filename, alg="", col_merge=[]):
    master_df = pd.read_csv(filename)

    print(f"Evaluate master df shape: {master_df.shape}")
    print(f"Nunique: {master_df.nunique().sort_values()}")

    if alg == "QuickGreedy":
        return run_quick(master_df, col_merge)

    return run_evolved(master_df, col_merge)


def evaluate(program_path):
    try:
        # Import the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if the required function exists
        if not hasattr(program, "Evolved"):
            return {
                "combined_score": 0.0,
                "runs_successfully": 0.0,
                "error": "Missing algorithm function",
            }

        # Test on different datasets
        test_files = ["datasets/movies.csv",
        "datasets/beer.csv",
        "datasets/BIRD.csv",
        "datasets/PDMX.csv",
        "datasets/products.csv"]

        col_merges = [
            [['movieinfo', 'movietitle', 'rottentomatoeslink']],
            [['beer/beerId', 'beer/name']],
            [['PostId', 'Body']],
            [['path', 'metadata'], ['hasmetadata', 'isofficial', 'isuserpublisher', 'isdraft', 'hasannotations', 'subsetall']],
            [['product_title', 'parent_asin']],
        ]

        failed_files = 0
        hit_rates = []
        total_runtime = 0.0

        for filename, col_merge in zip(test_files, col_merges):
            try:
                # Check if file exists
                if not os.path.exists(filename):
                    print(f"Dataset not found: {filename}, skipping...")
                    continue

                print(f"Processing dataset: {filename}")
                # This will test the algorithm with the dataset
                master_df = pd.read_csv(filename)
                st = time.time()
                reordered, _ = program.Evolved().reorder(
                    master_df,
                    early_stop=100000,
                    distinct_value_threshold=0.7,
                    row_stop=4,
                    col_stop=2,
                    col_merge=col_merge,
                )
                runtime = time.time() - st
                results = evaluate_df_prefix_hit_cnt(reordered)
                print(f"Results: {results}, Runtime: {runtime}")

                hit_rate = results[1] / 100

                hit_rates.append(hit_rate)
                total_runtime += runtime

            except Exception as e:
                print(f"Failed to process {os.path.basename(filename)}: {str(e)}")
                failed_files += 1
                break

        if failed_files > 0:
            return {
                "combined_score": 0.0,
                "runs_successfully": 0.0,
                "error": "1 or more files failed to run",
            }

        average_hit_rate = sum(hit_rates) / len(test_files)
        average_runtime = total_runtime / len(test_files)

        score = 0.95 * average_hit_rate + 0.05 * (12 - min(12, average_runtime)) / 12

        return {
            "combined_score": score,
            "runs_successfully": 1.0,
            "hit_rates": hit_rates,
            "total_runtime": total_runtime,
        }

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        print(traceback.format_exc())
        return {"combined_score": 0.0, "runs_successfully": 0.0, "error": str(e)}


if __name__ == "__main__":
    test_files = ["datasets/movies.csv",
        "datasets/beer.csv",
        "datasets/BIRD.csv",
        "datasets/PDMX.csv",
        "datasets/products.csv"]
    col_merges = [
        [['movieinfo', 'movietitle', 'rottentomatoeslink']],
        [['beer/beerId', 'beer/name']],
        [['PostId', 'Body']],
        [['path', 'metadata'], ['hasmetadata', 'isofficial', 'isuserpublisher', 'isdraft', 'hasannotations', 'subsetall']],
        [['product_title', 'parent_asin']],
    ]
    # Quick Greedy
    quick_hit_rates = []
    quick_runtimes = []
    for filename, col_merge in zip(test_files, col_merges):
        results, runtime = run(filename, "QuickGreedy", col_merge)
        print(results)
        print(runtime)
        hit_rate = results[1] / 100
        quick_hit_rates.append(hit_rate)
        quick_runtimes.append(runtime)
    # Baseline
    base_hit_rates = []
    base_runtimes = []
    for filename, col_merge in zip(test_files, col_merges):
        results, runtime = run(filename, "", col_merge)
        print(results)
        print(runtime)
        hit_rate = results[1] / 100
        base_hit_rates.append(hit_rate)
        base_runtimes.append(runtime)
    print("Quick Greedy hit rates:", quick_hit_rates)
    print("Quick Greedy runtimes:", quick_runtimes)
    quick_average_hit_rate = sum(quick_hit_rates) / len(test_files)
    quick_average_runtime = sum(quick_runtimes) / len(test_files)
    quick_score = 0.5 * quick_average_hit_rate + 0.5 * (
                1.0 / (1.0 + quick_average_runtime) if quick_average_runtime > 0 else 1.0
            )
    print("Quick Greedy Score:", quick_score)
    print("Baseline hit rates:", base_hit_rates)
    print("Baseline runtimes:", base_runtimes)
    base_average_hit_rate = sum(base_hit_rates) / len(test_files)
    base_average_runtime = sum(base_runtimes) / len(test_files)
    base_score = 0.5 * base_average_hit_rate + 0.5 * (
                1.0 / (1.0 + base_average_runtime) if base_average_runtime > 0 else 1.0
            )
    print("Baseline Score:", base_score)


```

## openevolve/examples/ADRS/llm_sql/initial_program.py

```python
import pandas as pd
from solver import Algorithm
from typing import Tuple, List
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache
from collections import Counter
import networkx as nx


class Evolved(Algorithm):
    """
    GGR algorithm
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

        self.dep_graph = None  # NOTE: not used, for one way dependency

        self.num_rows = 0
        self.num_cols = 0
        self.column_stats = None
        self.val_len = None
        self.row_stop = None
        self.col_stop = None
        self.base = 2000

    def find_max_group_value(self, df: pd.DataFrame, value_counts: Dict, early_stop: int = 0) -> str:
        # NOTE: recalculate value counts and length for each value
        value_counts = Counter(df.stack())
        weighted_counts = {val: self.val_len[val] * (count - 1) for val, count in value_counts.items()}  # if count > 1} TODO: why?
        if not weighted_counts:
            return None
        max_group_val, max_weighted_count = max(weighted_counts.items(), key=lambda x: x[1])
        if max_weighted_count < early_stop:
            return None
        return max_group_val

    def reorder_columns_for_value(self, row, value, column_names, grouped_rows_len: int = 1):
        # cols_with_value will now use attribute access instead of indexing with row[]
        cols_with_value = []
        for idx, col in enumerate(column_names):
            if hasattr(row, col) and getattr(row, col) == value:
                cols_with_value.append(col)
            elif hasattr(row, col.replace(" ", "_")) and getattr(row, col.replace(" ", "_")) == value:
                cols_with_value.append(col)
            else:
                attr_name = f"_{idx}"
                if hasattr(row, attr_name) and getattr(row, attr_name) == value:
                    cols_with_value.append(attr_name)

        if self.dep_graph is not None and grouped_rows_len > 1:
            # NOTE: experimental
            reordered_cols = []
            for col in cols_with_value:
                dependent_cols = self.get_dependent_columns(col)

                # check if dependent columns are in row, and if column exists in row attributes
                valid_dependent_cols = []
                for idx, dep_col in enumerate(dependent_cols):
                    if hasattr(row, dep_col):
                        valid_dependent_cols.append(dep_col)
                    elif hasattr(row, dep_col.replace(" ", "_")):
                        valid_dependent_cols.append(dep_col)
                    else:
                        attr_name = f"_{idx}"
                        if hasattr(row, attr_name):
                            valid_dependent_cols.append(dep_col)

                reordered_cols.extend([col] + valid_dependent_cols)
            cols_without_value = [col for col in column_names if col not in reordered_cols]
            reordered_cols.extend(cols_without_value)
            assert len(reordered_cols) == len(
                column_names
            ), f"Reordered cols len: {len(reordered_cols)}  Original cols len: {len(column_names)}"
            return [getattr(row, col) for col in reordered_cols], cols_with_value
        else:
            # NOTE: paper logic
            cols_without_value = []
            for idx, col in enumerate(column_names):
                if hasattr(row, col) and getattr(row, col) != value:
                    cols_without_value.append(col)
                elif hasattr(row, col.replace(" ", "_")) and getattr(row, col.replace(" ", "_")) != value:
                    cols_without_value.append(col)
                else:
                    # Handle some edge cases
                    attr_name = f"_{idx}"
                    if hasattr(row, attr_name) and getattr(row, attr_name) != value:
                        cols_without_value.append(attr_name)

            reordered_cols = cols_with_value + cols_without_value
            assert len(reordered_cols) == len(
                column_names
            ), f"Reordered cols len: {len(reordered_cols)}  Original cols len: {len(column_names)}"
            return [getattr(row, col) for col in reordered_cols], cols_with_value

    def get_dependent_columns(self, col: str) -> List[str]:
        if self.dep_graph is None or not self.dep_graph.has_node(col):
            return []
        return list(nx.descendants(self.dep_graph, col))

    @lru_cache(maxsize=None)
    def get_cached_dependent_columns(self, col: str) -> List[str]:
        return self.get_dependent_columns(col)

    def fixed_reorder(self, df: pd.DataFrame, row_sort: bool = True) -> Tuple[pd.DataFrame, List[List[str]]]:
        num_rows, column_stats = self.calculate_col_stats(df, enable_index=True)
        reordered_columns = [col for col, _, _, _ in column_stats]
        reordered_df = df[reordered_columns]

        assert reordered_df.shape == df.shape
        column_orderings = [reordered_columns] * num_rows

        if row_sort:
            reordered_df = reordered_df.sort_values(by=reordered_columns, axis=0)

        return reordered_df, column_orderings

    def column_recursion(self, result_df, max_value, grouped_rows, row_stop, col_stop, early_stop):
        cols_settled = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.reorder_columns_for_value, row, max_value, grouped_rows.columns.tolist(), len(grouped_rows))
                for row in grouped_rows.itertuples(index=False)
            ]
            for i, future in enumerate(as_completed(futures)):
                reordered_row, cols_settled = future.result()
                result_df.loc[i] = reordered_row

        grouped_value_counts = Counter()

        if not result_df.empty:
            # Group by the first column
            grouped_result_df = result_df.groupby(result_df.columns[0])
            grouped_value_counts = Counter(grouped_rows.stack())  # this is still faster than updating from cached value counts

            for _, group in grouped_result_df:
                if group[group.columns[0]].iloc[0] != max_value:
                    continue

                dependent_cols = self.get_cached_dependent_columns(group.columns[0])
                length_of_settle_cols = len(cols_settled)

                if dependent_cols:
                    assert length_of_settle_cols >= 1, f"Dependent columns should be no less than 1, but got {length_of_settle_cols}"

                    # test the first length_of_settle_cols columns, each column has nunique == 1
                    for col in group.columns[:length_of_settle_cols]:
                        assert group[col].nunique() == 1, f"Column {col} should have nunique == 1, but got {group[col].nunique()}"

                    # drop all the settled columns and reorder the rest
                    group_remainder = group.iloc[:, length_of_settle_cols:]
                else:
                    group_remainder = group.iloc[:, 1:]

                grouped_remainder_value_counts = Counter(group_remainder.stack())

                reordered_group_remainder, _ = self.recursive_reorder(
                    group_remainder, grouped_remainder_value_counts, early_stop=early_stop, row_stop=row_stop, col_stop=col_stop + 1
                )
                # Update the group with the reordered columns
                if dependent_cols:
                    group.iloc[:, length_of_settle_cols:] = reordered_group_remainder.values
                else:
                    group.iloc[:, 1:] = reordered_group_remainder.values

                result_df.update(group)
                break

        return result_df, grouped_value_counts

    def recursive_reorder(
        self,
        df: pd.DataFrame,
        value_counts: Dict,
        early_stop: int = 0,
        original_columns: List[str] = None,
        row_stop: int = 0,
        col_stop: int = 0,
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        if df.empty or len(df.columns) == 0 or len(df) == 0:
            return df, []

        if self.row_stop is not None and row_stop >= self.row_stop:
            return self.fixed_reorder(df)

        if self.col_stop is not None and col_stop >= self.col_stop:
            return self.fixed_reorder(df)

        if original_columns is None:
            original_columns = df.columns.tolist()

        # Find the max group value using updated counts
        max_value = self.find_max_group_value(df, value_counts, early_stop=early_stop)
        if max_value is None:
            # If there is no max value, then fall back to fixed reorder
            return self.fixed_reorder(df)

        grouped_rows = df[df.isin([max_value]).any(axis=1)]
        remaining_rows = df[~df.isin([max_value]).any(axis=1)]

        # If there is no grouped rows, return the original DataFrame
        if grouped_rows.empty:
            return self.fixed_reorder(df)

        result_df = pd.DataFrame(columns=df.columns)

        reordered_remaining_rows = pd.DataFrame(columns=df.columns)  # Initialize empty dataframe first

        # Column Recursion
        result_df, grouped_value_counts = self.column_recursion(result_df, max_value, grouped_rows, row_stop, col_stop, early_stop)

        remaining_value_counts = value_counts - grouped_value_counts  # Approach 1 - update remaining value counts with subtraction

        # Row Recursion
        reordered_remaining_rows, _ = self.recursive_reorder(
            remaining_rows, remaining_value_counts, early_stop=early_stop, row_stop=row_stop + 1, col_stop=col_stop
        )
        old_column_names = result_df.columns.tolist()
        result_cols_reset = result_df.reset_index(drop=True)
        result_rows_reset = reordered_remaining_rows.reset_index(drop=True)
        final_result_df = pd.DataFrame(result_cols_reset.values.tolist() + result_rows_reset.values.tolist())

        if row_stop == 0 and col_stop == 0:
            final_result_df.columns = old_column_names
            final_result_df.columns = final_result_df.columns.tolist()[:-1] + ["original_index"]

        return final_result_df, []

    def recursive_split_and_reorder(self, df: pd.DataFrame, original_columns: List[str] = None, early_stop: int = 0):
        """
        Recursively split the DataFrame into halves until the size is <= 1000, then apply the recursive reorder function.
        """
        if len(df) <= self.base:
            initial_value_counts = Counter(df.stack())
            return self.recursive_reorder(df, initial_value_counts, early_stop, original_columns, row_stop=0, col_stop=0)[0]

        mid_index = len(df) // 2
        df_top_half = df.iloc[:mid_index]
        df_bottom_half = df.iloc[mid_index:]

        with ProcessPoolExecutor() as executor:
            future_top = executor.submit(self.recursive_split_and_reorder, df_top_half, original_columns, early_stop)
            future_bottom = executor.submit(self.recursive_split_and_reorder, df_bottom_half, original_columns, early_stop)

        reordered_top_half = future_top.result()
        reordered_bottom_half = future_bottom.result()

        assert reordered_bottom_half.shape == df_bottom_half.shape
        reordered_df = pd.concat([reordered_top_half, reordered_bottom_half], axis=0, ignore_index=True)

        assert reordered_df.shape == df.shape

        return reordered_df

    @lru_cache(maxsize=None)
    def calculate_length(self, value):
        if isinstance(value, bool):
            return 4**2
        if isinstance(value, (int, float)):
            return len(str(value)) ** 2
        if isinstance(value, str):
            return len(value) ** 2
        return 0

    def reorder(
        self,
        df: pd.DataFrame,
        early_stop: int = 0,
        row_stop: int = None,
        col_stop: int = None,
        col_merge: List[List[str]] = [],
        one_way_dep: List[Tuple[str, str]] = [],
        distinct_value_threshold: float = 0.8,
        parallel: bool = True,
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        # Prepare
        initial_df = df.copy()
        if col_merge:
            self.num_rows, self.column_stats = self.calculate_col_stats(df, enable_index=True)
            reordered_columns = [col for col, _, _, _ in self.column_stats]
            for col_to_merge in col_merge:
                final_col_order = [col for col in reordered_columns if col in col_to_merge]
                df = self.merging_columns(df, final_col_order, prepended=False)
        self.num_rows, self.column_stats = self.calculate_col_stats(df, enable_index=True)
        self.column_stats = {col: (num_groups, avg_len, score) for col, num_groups, avg_len, score in self.column_stats}

        # One way dependency statistics [not used]
        if one_way_dep is not None and len(one_way_dep) > 0:
            self.dep_graph = nx.DiGraph()
            for dep in one_way_dep:
                col1 = [col for col in df.columns if dep[0] in col]
                col2 = [col for col in df.columns if dep[1] in col]
                assert len(col1) == 1, f"Expected one column to match {dep[0]}, but got {len(col1)}"
                assert len(col2) == 1, f"Expected one column to match {dep[1]}, but got {len(col2)}"
                col1 = col1[0]
                col2 = col2[0]
                self.dep_graph.add_edge(col1, col2)

        # Discard too distinct columns by threshold [optional]
        nunique_threshold = len(df) * distinct_value_threshold
        columns_to_discard = [col for col in df.columns if df[col].nunique() > nunique_threshold]
        columns_to_discard = sorted(columns_to_discard, key=lambda x: self.column_stats[x][2], reverse=True)
        columns_to_recurse = [col for col in df.columns if col not in columns_to_discard]
        df["original_index"] = range(len(df))
        discarded_columns_df = df[columns_to_discard + ["original_index"]]
        df_to_recurse = df[columns_to_recurse + ["original_index"]]
        recurse_df = df_to_recurse

        self.column_stats = {col: stats for col, stats in self.column_stats.items() if col not in columns_to_discard}
        initial_value_counts = Counter(recurse_df.stack())
        self.val_len = {val: self.calculate_length(val) for val in initial_value_counts.keys()}

        self.row_stop = row_stop if row_stop else len(recurse_df)
        self.col_stop = col_stop if col_stop else len(recurse_df.columns.tolist())
        print("*" * 80)
        print(f"DF columns = {df.columns}")
        # print(f"Early stop = {early_stop}")
        # print(f"Row recursion stop depth = {self.row_stop}, Column recursion stop depth = {self.col_stop}")
        print("*" * 80)

        # Eary stop and fall back
        recurse_df, _ = self.fixed_reorder(recurse_df)

        # Recursive reordering
        self.num_cols = len(recurse_df.columns)
        if parallel:
            reordered_df = self.recursive_split_and_reorder(recurse_df, original_columns=columns_to_recurse, early_stop=early_stop)
        else:
            reordered_df, _ = self.recursive_reorder(
                recurse_df,
                initial_value_counts,
                early_stop=early_stop,
            )

        assert (
            reordered_df.shape == recurse_df.shape
        ), f"Reordered DataFrame shape {reordered_df.shape} does not match original DataFrame shape {recurse_df.shape}"
        assert recurse_df["original_index"].is_unique, "Passed in recurse index contains duplicates!"
        assert reordered_df["original_index"].is_unique, "Reordered index contains duplicates!"

        if len(columns_to_discard) > 0:
            final_df = pd.merge(reordered_df, discarded_columns_df, on="original_index", how="left")
        else:
            final_df = reordered_df

        final_df = final_df.drop(columns=["original_index"])

        if not col_merge:
            assert (
                final_df.shape == initial_df.shape
            ), f"Final DataFrame shape {final_df.shape} does not match original DataFrame shape {initial_df.shape}"
        else:
            assert (
                final_df.shape[0] == initial_df.shape[0]
            ), f"Final DataFrame shape {final_df.shape} does not match original DataFrame shape {initial_df.shape}"
            assert (
                final_df.shape[1] == recurse_df.shape[1] + len(columns_to_discard) - 1
            ), f"Final DataFrame shape {final_df.shape} does not match original DataFrame shape {recurse_df.shape}"

        # sort by the first column to get the final order
        final_df = final_df.sort_values(by=final_df.columns.to_list(), axis=0)
        return final_df, []
```

## openevolve/examples/ADRS/llm_sql/quick_greedy.py

```python
import pandas as pd
from solver import Algorithm
from typing import Tuple, List
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache
from collections import Counter
import networkx as nx

class QuickGreedy(Algorithm):
    """
    Select the group with the highest score, recurse

    - Score = length of the group value * (number of rows with the group value - 1)
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

        # For one way dependency
        self.dep_graph = None

        # For fixed column fallback
        self.num_rows = 0
        self.num_cols = 0
        self.column_stats = None
        self.val_len = None
        self.row_stop = None
        self.col_stop = None
        self.base = 2000

    def find_max_group_value(self, df: pd.DataFrame, value_counts: Dict, early_stop: int = 0) -> str:
        value_counts = Counter(df.stack())
        weighted_counts = {val: self.val_len[val] * (count - 1) for val, count in value_counts.items()}  # if count > 1} TODO: why?
        if not weighted_counts:
            return None
        max_group_val, max_weighted_count = max(weighted_counts.items(), key=lambda x: x[1])
        if max_weighted_count < early_stop:
            return None
        return max_group_val

    def reorder_columns_for_value(self, row, value, column_names, grouped_rows_len: int = 1):
        cols_with_value = []
        for idx, col in enumerate(column_names):
            if hasattr(row, col) and getattr(row, col) == value:
                cols_with_value.append(col)
            elif hasattr(row, col.replace(" ", "_")) and getattr(row, col.replace(" ", "_")) == value:
                cols_with_value.append(col)
            else:
                attr_name = f"_{idx}"
                if hasattr(row, attr_name) and getattr(row, attr_name) == value:
                    cols_with_value.append(attr_name)

        if self.dep_graph is not None and grouped_rows_len > 1:
            reordered_cols = []
            for col in cols_with_value:
                dependent_cols = self.get_dependent_columns(col)

                valid_dependent_cols = []
                for idx, dep_col in enumerate(dependent_cols):
                    if hasattr(row, dep_col):
                        valid_dependent_cols.append(dep_col)
                    elif hasattr(row, dep_col.replace(" ", "_")):
                        valid_dependent_cols.append(dep_col)
                    else:
                        attr_name = f"_{idx}"
                        if hasattr(row, attr_name):
                            valid_dependent_cols.append(dep_col)

                reordered_cols.extend([col] + valid_dependent_cols)
            cols_without_value = [col for col in column_names if col not in reordered_cols]
            reordered_cols.extend(cols_without_value)
            assert len(reordered_cols) == len(
                column_names
            ), f"Reordered cols len: {len(reordered_cols)}  Original cols len: {len(column_names)}"

            return [getattr(row, col) for col in reordered_cols], cols_with_value
        else:
            cols_without_value = []
            for idx, col in enumerate(column_names):
                if hasattr(row, col) and getattr(row, col) != value:
                    cols_without_value.append(col)
                elif hasattr(row, col.replace(" ", "_")) and getattr(row, col.replace(" ", "_")) != value:
                    cols_without_value.append(col)
                else:
                    # strange case where the column is not in the row attributes, represented in _0, _1, _2, etc.
                    attr_name = f"_{idx}"
                    if hasattr(row, attr_name) and getattr(row, attr_name) != value:
                        cols_without_value.append(attr_name)

            reordered_cols = cols_with_value + cols_without_value

            assert len(reordered_cols) == len(
                column_names
            ), f"Reordered cols len: {len(reordered_cols)}  Original cols len: {len(column_names)}"
            # Return reordered columns and cols_with_value
            return [getattr(row, col) for col in reordered_cols], cols_with_value

    def get_dependent_columns(self, col: str) -> List[str]:
        """Get all columns that depend on the given column."""
        if self.dep_graph is None or not self.dep_graph.has_node(col):
            return []
        return list(nx.descendants(self.dep_graph, col))

    @lru_cache(maxsize=None)
    def get_cached_dependent_columns(self, col: str) -> List[str]:
        return self.get_dependent_columns(col)

    def fixed_reorder(self, df: pd.DataFrame, row_sort: bool = True) -> Tuple[pd.DataFrame, List[List[str]]]:
        num_rows, column_stats = self.calculate_col_stats(df, enable_index=True)
        reordered_columns = [col for col, _, _, _ in column_stats]
        reordered_df = df[reordered_columns]

        assert reordered_df.shape == df.shape
        column_orderings = [reordered_columns] * num_rows

        if row_sort:
            reordered_df = reordered_df.sort_values(by=reordered_columns, axis=0)

        return reordered_df, column_orderings

    def column_recursion(self, result_df, max_value, grouped_rows, row_stop, col_stop, early_stop):
        cols_settled = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.reorder_columns_for_value, row, max_value, grouped_rows.columns.tolist(), len(grouped_rows))
                for row in grouped_rows.itertuples(index=False)
            ]
            for i, future in enumerate(as_completed(futures)):
                reordered_row, cols_settled = future.result()
                result_df.loc[i] = reordered_row

        grouped_value_counts = Counter()

        if not result_df.empty:
            grouped_result_df = result_df.groupby(result_df.columns[0])
            grouped_value_counts = Counter(grouped_rows.stack())  # this is still faster than updating from cached value counts

            for _, group in grouped_result_df:
                if group[group.columns[0]].iloc[0] != max_value:
                    continue

                dependent_cols = self.get_cached_dependent_columns(group.columns[0])
                length_of_settle_cols = len(cols_settled)

                if dependent_cols:
                    assert length_of_settle_cols >= 1, f"Dependent columns should be no less than 1, but got {length_of_settle_cols}"

                    for col in group.columns[:length_of_settle_cols]:
                        assert group[col].nunique() == 1, f"Column {col} should have nunique == 1, but got {group[col].nunique()}"

                    group_remainder = group.iloc[:, length_of_settle_cols:]
                else:
                    group_remainder = group.iloc[:, 1:]

                grouped_remainder_value_counts = Counter(group_remainder.stack())

                reordered_group_remainder, _ = self.recursive_reorder(
                    group_remainder, grouped_remainder_value_counts, early_stop=early_stop, row_stop=row_stop, col_stop=col_stop + 1
                )
                if dependent_cols:
                    group.iloc[:, length_of_settle_cols:] = reordered_group_remainder.values
                else:
                    group.iloc[:, 1:] = reordered_group_remainder.values

                result_df.update(group)
                break

        return result_df, grouped_value_counts

    def recursive_reorder(
        self,
        df: pd.DataFrame,
        value_counts: Dict,
        early_stop: int = 0,
        original_columns: List[str] = None,
        row_stop: int = 0,
        col_stop: int = 0,
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        if df.empty or len(df.columns) == 0 or len(df) == 0:
            return df, []

        if self.row_stop is not None and row_stop >= self.row_stop:
            return self.fixed_reorder(df)

        if self.col_stop is not None and col_stop >= self.col_stop:
            return self.fixed_reorder(df)

        if original_columns is None:
            original_columns = df.columns.tolist()

        max_value = self.find_max_group_value(df, value_counts, early_stop=early_stop)
        if max_value is None:
            return self.fixed_reorder(df)

        grouped_rows = df[df.isin([max_value]).any(axis=1)]
        remaining_rows = df[~df.isin([max_value]).any(axis=1)]

        if grouped_rows.empty:
            return self.fixed_reorder(df)

        result_df = pd.DataFrame(columns=df.columns)

        reordered_remaining_rows = pd.DataFrame(columns=df.columns)  # Initialize empty dataframe first

        result_df, grouped_value_counts = self.column_recursion(result_df, max_value, grouped_rows, row_stop, col_stop, early_stop)

        remaining_value_counts = value_counts - grouped_value_counts  # Approach 1 - update remaining value counts with subtraction

        reordered_remaining_rows, _ = self.recursive_reorder(
            remaining_rows, remaining_value_counts, early_stop=early_stop, row_stop=row_stop + 1, col_stop=col_stop
        )

        old_column_names = result_df.columns.tolist()
        result_cols_reset = result_df.reset_index(drop=True)
        result_rows_reset = reordered_remaining_rows.reset_index(drop=True)

        final_result_df = pd.DataFrame(result_cols_reset.values.tolist() + result_rows_reset.values.tolist())

        if row_stop == 0 and col_stop == 0:
            final_result_df.columns = old_column_names

            final_result_df.columns = final_result_df.columns.tolist()[:-1] + ["original_index"]

        return final_result_df, []

    def recursive_split_and_reorder(
        self, df: pd.DataFrame, original_columns: List[str] = None, early_stop: int = 0
    ):
        """
        Recursively split the DataFrame into halves until the size is <= 1000, then apply the recursive reorder function.
        """

        if len(df) <= self.base:
            initial_value_counts = Counter(df.stack())
            return self.recursive_reorder(df, initial_value_counts, early_stop, original_columns, row_stop=0, col_stop=0)[0]

        mid_index = len(df) // 2
        df_top_half = df.iloc[:mid_index]
        df_bottom_half = df.iloc[mid_index:]

        with ProcessPoolExecutor() as executor:
            future_top = executor.submit(self.recursive_split_and_reorder, df_top_half, original_columns, early_stop)
            future_bottom = executor.submit(
                self.recursive_split_and_reorder, df_bottom_half, original_columns, early_stop
            )

        reordered_top_half = future_top.result()
        reordered_bottom_half = future_bottom.result()

        assert reordered_bottom_half.shape == df_bottom_half.shape

        reordered_df = pd.concat([reordered_top_half, reordered_bottom_half], axis=0, ignore_index=True)

        assert reordered_df.shape == df.shape

        return reordered_df

    @lru_cache(maxsize=None)
    def calculate_length(self, value):
        if isinstance(value, bool):
            return 4**2
        if isinstance(value, (int, float)):
            return len(str(value)) ** 2
        if isinstance(value, str):
            return len(value) ** 2
        return 0

    def reorder(
        self,
        df: pd.DataFrame,
        early_stop: int = 0,
        row_stop: int = None,
        col_stop: int = None,
        col_merge: List[List[str]] = [],
        one_way_dep: List[Tuple[str, str]] = [],
        distinct_value_threshold: float = 0.8,
        parallel: bool = True,
        # parallel: bool = False,
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        initial_df = df.copy()
        if col_merge:
            self.num_rows, self.column_stats = self.calculate_col_stats(df, enable_index=True)
            reordered_columns = [col for col, _, _, _ in self.column_stats]
            for col_to_merge in col_merge:
                final_col_order = [col for col in reordered_columns if col in col_to_merge]
                df = self.merging_columns(df, final_col_order, prepended=False)  # NOTE: prepend should always be false for merging columns

        self.num_rows, self.column_stats = self.calculate_col_stats(df, enable_index=True)
        self.column_stats = {col: (num_groups, avg_len, score) for col, num_groups, avg_len, score in self.column_stats}

        if one_way_dep is not None and len(one_way_dep) > 0:
            self.dep_graph = nx.DiGraph()
            for dep in one_way_dep:
                col1 = [col for col in df.columns if dep[0] in col]
                col2 = [col for col in df.columns if dep[1] in col]
                assert len(col1) == 1, f"Expected one column to match {dep[0]}, but got {len(col1)}"
                assert len(col2) == 1, f"Expected one column to match {dep[1]}, but got {len(col2)}"
                col1 = col1[0]
                col2 = col2[0]
                self.dep_graph.add_edge(col1, col2)

        nunique_threshold = len(df) * distinct_value_threshold
        columns_to_discard = [col for col in df.columns if df[col].nunique() > nunique_threshold]
        columns_to_discard = sorted(columns_to_discard, key=lambda x: self.column_stats[x][2], reverse=True)  # sort based on scores
        columns_to_recurse = [col for col in df.columns if col not in columns_to_discard]
        print(f"Discarding columns: {columns_to_discard}")
        df["original_index"] = range(len(df))

        discarded_columns_df = df[columns_to_discard + ["original_index"]]
        df_to_recurse = df[columns_to_recurse + ["original_index"]]
        recurse_df = df_to_recurse

        self.column_stats = {col: stats for col, stats in self.column_stats.items() if col not in columns_to_discard}
        initial_value_counts = Counter(recurse_df.stack())

        # Cache the length of each value
        self.val_len = {val: self.calculate_length(val) for val in initial_value_counts.keys()}

        self.row_stop = row_stop if row_stop else len(recurse_df)
        self.col_stop = col_stop if col_stop else len(recurse_df.columns.tolist())
        print("*" * 80)
        print(f"DF columns = {df.columns}")
        print(f"Early stop = {early_stop}")
        print(f"Row recursion stop depth = {self.row_stop}, Column recursion stop depth = {self.col_stop}")
        print("*" * 80)

        recurse_df, _ = self.fixed_reorder(recurse_df)

        # Recursive reordering
        self.num_cols = len(recurse_df.columns)
        if parallel:
            reordered_df = self.recursive_split_and_reorder(
                recurse_df,
                original_columns=columns_to_recurse,
                early_stop=early_stop
            )
        else:
            reordered_df, column_orders = self.recursive_reorder(
                recurse_df,
                initial_value_counts,
                early_stop=early_stop,
            )

        assert (
            reordered_df.shape == recurse_df.shape
        ), f"Reordered DataFrame shape {reordered_df.shape} does not match original DataFrame shape {recurse_df.shape}"
        assert recurse_df["original_index"].is_unique, "Passed in recurse index contains duplicates!"
        assert reordered_df["original_index"].is_unique, "Reordered index contains duplicates!"

        if len(columns_to_discard) > 0:
            final_df = pd.merge(reordered_df, discarded_columns_df, on="original_index", how="left")
        else:
            final_df = reordered_df

        print(
            f"Reordered df shape: {reordered_df.shape}, discarded columns df shape: {discarded_columns_df.shape}, final df shape: {final_df.shape}"
        )

        final_df = final_df.drop(columns=["original_index"])

        if not col_merge:
            assert (
                final_df.shape == initial_df.shape
            ), f"Final DataFrame shape {final_df.shape} does not match original DataFrame shape {initial_df.shape}"
        else:
            assert (
                final_df.shape[0] == initial_df.shape[0]
            ), f"Final DataFrame shape {final_df.shape} does not match original DataFrame shape {initial_df.shape}"
            assert (
                final_df.shape[1] == recurse_df.shape[1] + len(columns_to_discard) - 1
            ), f"Final DataFrame shape {final_df.shape} does not match original DataFrame shape {recurse_df.shape}"

        final_df = final_df.sort_values(by=final_df.columns.to_list(), axis=0)
        return final_df, []
```

## openevolve/examples/ADRS/llm_sql/solver.py

```python
import pandas as pd
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from utils import Trie
import time


class Algorithm:
    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    def reorder(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Subclasses should implement this!")

    @staticmethod
    def evaluate_df_prefix_hit_cnt(self, df: pd.DataFrame) -> int:
        """
        Function to evaluate the prefix hit count of a DataFrame
        """

        def max_overlap(trie, row_string):
            return trie.longest_common_prefix(row_string)

        trie = Trie()
        total_prefix_hit_count = 0

        def process_row(index, row):
            row_string = "".join(row.astype(str).values)  # No spaces between columns
            row_prefix_hit_count = max_overlap(trie, row_string)
            trie.insert(row_string)
            return row_prefix_hit_count

        with ThreadPoolExecutor() as executor:
            results = executor.map(process_row, df.index, [row for _, row in df.iterrows()])

        total_prefix_hit_count = sum(results)
        return total_prefix_hit_count

    @staticmethod
    def evaluate_cell_hit_cnt(df: pd.DataFrame) -> int:
        """
        Function to evaluate the prefix hit count of a DataFrame based on exact cell matching.
        For a cell to be a hit, all previous cells in the row must also be hits.
        """

        total_prefix_hit_count = 0
        seen_rows = set()  # Cache of fully processed rows

        def process_row(index, row):
            nonlocal seen_rows
            prefix_hit_count = 0
            current_row_cache = []

            for col_value in row:
                # Check if adding this cell matches exactly with prior cache
                current_row_cache.append(col_value)
                if tuple(current_row_cache) in seen_rows:
                    prefix_hit_count += 1
                else:
                    break  # Stop counting hits if any cell isn't in the cache

            seen_rows.add(tuple(row))  # Add the fully processed row to cache
            return prefix_hit_count

        # Process each row sequentially (row-to-row comparison for hits)
        for _, row in df.iterrows():
            total_prefix_hit_count += process_row(_, row)

        return total_prefix_hit_count

    @staticmethod
    def get_groups_values(df: pd.DataFrame):
        """
        Function to get the value counts of a DataFrame
        """
        if df.empty:
            return {}
        value_counts = df.stack().value_counts()
        if value_counts.empty:
            return {}
        return value_counts

    @staticmethod
    def calculate_length(value):
        val = 0
        if isinstance(value, bool):
            val = 4  # length of 'True' or 'False'
        elif isinstance(value, (int, float)):
            val = len(str(value))
        elif isinstance(value, str):
            val = len(value)
        else:
            val = 0
        return val**2

    @staticmethod
    def drop_col(df: pd.DataFrame, col):
        return df.drop(columns=[col])

    @staticmethod
    def drop_rows(df: pd.DataFrame, rows):
        return df.drop(index=rows)

    @staticmethod
    def merging_columns(df: pd.DataFrame, col_names: List[str], delimiter: str = "_", prepended: bool = False) -> pd.DataFrame:
        if not all(col in df.columns for col in col_names):
            raise ValueError("Column names not found in DataFrame")

        # before merging, check that each column to be merged has the same number of unique values
        if len(set(df[col_names].nunique())) != 1:
            raise ValueError(f"Columns to be merged {col_names}, do not have the same number of unique values: {df.nunique().sort_values()}")

        merged_names = delimiter.join(col_names)
        if prepended:
            df[merged_names] = df[col_names].apply(
                lambda x: merged_names + ": " + delimiter.join([val.split(": ", 1)[1] for col, val in zip(col_names, x)]), axis=1
            )
        else:
            df[merged_names] = df[col_names].apply(lambda x: "".join([f"{val}" for val in x]), axis=1)
        df = df.drop(columns=col_names)
        return df

    @staticmethod
    def calculate_col_stats(df: pd.DataFrame, enable_index=False):
        num_rows = len(df)
        column_stats = []
        for col in df.columns:
            if col == "original_index":
                continue

            num_groups = df[col].nunique()
            if df[col].dtype == "object" or df[col].dtype == "string":
                avg_length = df[col].astype(str).str.len().mean()
            elif df[col].dtype == "bool":
                avg_length = 4  # Assuming 'True' or 'False' as average length
            elif df[col].dtype in ["int64", "float64"]:
                avg_length = df[col].astype(str).str.len().mean()
            else:
                avg_length = 0

            avg_length = avg_length**2

            if num_groups == 0:
                score = 0
            else:
                # Average size per group: number of rows in each group
                avg_size_per_group = num_rows / num_groups
                # score = avg_size_per_group * avg_length
                score = avg_length * (avg_size_per_group - 1)

                if num_rows == num_groups:  # no sharing at all
                    score = 0
            column_stats.append((col, num_groups, avg_length, score))

        # original_index all distinct values, so give lowest score
        if enable_index and "original_index" in df.columns:
            column_stats.append(("original_index", len(df), 0, 0))

        # Sort the columns based on the score
        column_stats.sort(key=lambda x: x[3], reverse=True)
        return num_rows, column_stats

```

## openevolve/examples/ADRS/llm_sql/utils.py

```python
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from typing import List, Tuple
from pyspark.sql import SparkSession
from itertools import combinations
import yaml

class TrieNode:
    def __init__(self):
        self.children = {}
        self.end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.end_of_word = True

    def longest_common_prefix(self, word):
        node = self.root
        common_prefix_length = 0
        for char in word:
            if char in node.children:
                common_prefix_length += len(char)
                node = node.children[char]
            else:
                break
        return common_prefix_length

def calculate_length(value):
    val = 0
    if isinstance(value, bool):
        val = 4  # length of 'True' or 'False'
    elif isinstance(value, (int, float)):
        val = len(str(value))
    elif isinstance(value, str):
        val = len(value)
    else:
        val = 0
    return val**2

def evaluate_df_prefix_hit_cnt(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Function to evaluate the prefix hit count of a DataFrame
    """

    def max_overlap(trie, row_string):
        return min(len(row_string), trie.longest_common_prefix(row_string))


    trie = Trie()
    total_prefix_hit_count = 0
    total_string_length = 0

    def process_row(index, row):
        nonlocal total_string_length
        row_string = "".join(row.fillna("").astype(str).values)  # No spaces between columns
        total_string_length += len(row_string)
        row_prefix_hit_count = max_overlap(trie, row_string)
        trie.insert(row_string)
        return row_prefix_hit_count

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_row, df.index, [row for _, row in df.iterrows()])

    total_prefix_hit_count = sum(results)
    total_prefix_hit_rate = total_prefix_hit_count / total_string_length
    assert total_prefix_hit_count <= total_string_length
    print(f"Total string length: {total_string_length}")
    # NOTE: GPT-4o
    # /5: convert string length to token (rough estimation)
    no_cache_pricing = 2.5 / 5  # per 1M if not cached
    cache_pricing = 1.25 / 5  # per 1M if cached
    cached_tokens_pricing = total_prefix_hit_count * cache_pricing / 1e6
    non_cached_tokens_pricing = (total_string_length - total_prefix_hit_count) * no_cache_pricing / 1e6
    print(
        f"Cached tokens pricing = {round(cached_tokens_pricing,2)}, Non-cached tokens pricing = {round(non_cached_tokens_pricing,2)}, total pricing = {round(cached_tokens_pricing + non_cached_tokens_pricing,2)}"
    )
    return total_prefix_hit_count, total_prefix_hit_rate * 100
```

- https://spec.commonmark.org ```
  Markdown is a plain text format for writing structured documents, based on conventions for indicating formatting in email and usenet posts. It was developed by John Gruber (with help from Aaron Swartz)...

      The overriding design goal for Markdown’s formatting syntax is to make it as readable as possible. The idea is that a Markdown-formatted document should be publishable as-is, as plain text, without looking like it’s been marked up with tags or formatting instructions. (https://daringfireball.net/projects/markdown/)

The point can be illustrated by comparing a sample of AsciiDoc with an equivalent sample of Markdown. Here is a sample of AsciiDoc from the AsciiDoc manual:

1. List item one.

- List item one continued with a second paragraph followed by an
  Indented block.
- .................
  $ ls _.sh
  $ mv _.sh ~/tmp
  .................
- List item continued with a third paragraph.

2. List item two continued with an open block.

---

This paragraph is part of the preceding list item.

a. This list is nested and does not require explicit item
continuation.

- This paragraph is part of the preceding list item.

b. List item b.

## This paragraph belongs to item two of the outer list.

And here is the equivalent in Markdown:

1.  List item one.

    List item one continued with a second paragraph followed by an
    Indented block.

        $ ls *.sh
        $ mv *.sh ~/tmp

    List item continued with a third paragraph.

2.  List item two continued with an open block.

    This paragraph is part of the preceding list item.

    1. This list is nested and does not require explicit item continuation.

       This paragraph is part of the preceding list item.

    2. List item b.

    This paragraph belongs to item two of the outer list.

The AsciiDoc version is, arguably, easier to write. You don’t need to worry about indentation. But the Markdown version is much easier to read. The nesting of list items is apparent to the eye in the source, not just in the processed document.
1.2Why is a spec needed?

John Gruber’s canonical description of Markdown’s syntax does not specify the syntax unambiguously. Here are some examples of questions it does not answer:

    How much indentation is needed for a sublist? The spec says that continuation paragraphs need to be indented four spaces, but is not fully explicit about sublists. It is natural to think that they, too, must be indented four spaces, but Markdown.pl does not require that. This is hardly a “corner case,” and divergences between implementations on this issue often lead to surprises for users in real documents. (See this comment by John Gruber.)

    Is a blank line needed before a block quote or heading? Most implementations do not require the blank line. However, this can lead to unexpected results in hard-wrapped text, and also to ambiguities in parsing (note that some implementations put the heading inside the blockquote, while others do not). (John Gruber has also spoken in favor of requiring the blank lines.)

    Is a blank line needed before an indented code block? (Markdown.pl requires it, but this is not mentioned in the documentation, and some implementations do not require it.)

    paragraph
        code?

    What is the exact rule for determining when list items get wrapped in <p> tags? Can a list be partially “loose” and partially “tight”? What should we do with a list like this?

    1. one

    2. two
    3. three

    Or this?

    1.  one
        - a

        - b
    2.  two

    (There are some relevant comments by John Gruber here.)

    Can list markers be indented? Can ordered list markers be right-aligned?

     8. item 1
     9. item 2
    10. item 2a

    Is this one list with a thematic break in its second item, or two lists separated by a thematic break?

    * a
    * * * * *
    * b

    When list markers change from numbers to bullets, do we have two lists or one? (The Markdown syntax description suggests two, but the perl scripts and many other implementations produce one.)

    1. fee
    2. fie
    -  foe
    -  fum

    What are the precedence rules for the markers of inline structure? For example, is the following a valid link, or does the code span take precedence ?

    [a backtick (`)](/url) and [another backtick (`)](/url).

    What are the precedence rules for markers of emphasis and strong emphasis? For example, how should the following be parsed?

    *foo *bar* baz*

    What are the precedence rules between block-level and inline-level structure? For example, how should the following be parsed?

    - `a long code span can contain a hyphen like this
      - and it can screw things up`

    Can list items include section headings? (Markdown.pl does not allow this, but does allow blockquotes to include headings.)

    - # Heading

    Can list items be empty?

    * a
    *
    * b

    Can link references be defined inside block quotes or list items?

    > Blockquote [foo].
    >
    > [foo]: /url

    If there are multiple definitions for the same reference, which takes precedence?

    [foo]: /url1
    [foo]: /url2

    [foo][]

In the absence of a spec, early implementers consulted Markdown.pl to resolve these ambiguities. But Markdown.pl was quite buggy, and gave manifestly bad results in many cases, so it was not a satisfactory replacement for a spec.

Because there is no unambiguous spec, implementations have diverged considerably. As a result, users are often surprised to find that a document that renders one way on one system (say, a GitHub wiki) renders differently on another (say, converting to docbook using pandoc). To make matters worse, because nothing in Markdown counts as a “syntax error,” the divergence often isn’t discovered right away.

---

---

---

# Generated Evolution Spaces by Prompts

0.

1.

2.  https://chatgpt.com/c/690eb31b-974c-8331-912e-51f2ab4a2cf2

3.  [jerlendds](https://github.com/jerlendds) created this one!
