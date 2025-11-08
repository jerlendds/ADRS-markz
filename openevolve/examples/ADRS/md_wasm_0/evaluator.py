"""
Evaluator for the OpenEvolve Markdown→HTML task.

It imports the `initial_program.py` module (selected by program_path), calls
materialize_crate(...) to create a Rust project, builds native + wasm32-wasi,
then evaluates:
  1) CommonMark correctness on a canonical-ish subset we embed here
  2) Throughput/latency on a synthetic but realistic doc
  3) WASM code size

Combined score in [0,1]:
  score = 0.7 * correctness + 0.25 * perf + 0.05 * size

Hard failures:
  - Build failures or runtime errors → score = 0

Requirements in environment (best effort):
  - rustc + cargo
  - wasm32-wasi target: `rustup target add wasm32-wasi`
  - wasmtime (preferred) or wasmer; if neither present, we still score native
    but cap the total by 0.6 to incentivize WASM success.
"""

import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ------------ utilities ------------

def _which(cmd: str) -> str | None:
    return shutil.which(cmd)

def _run(cmd, cwd=None, input_bytes=None, timeout=60):
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdin=subprocess.PIPE if input_bytes is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    try:
        out, err = proc.communicate(input=input_bytes, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return 124, b"", b"TIMEOUT"
    return proc.returncode, out, err

# Small but diverse CommonMark subset (hand-picked from the spec and common cases)
# Each case: (markdown, expected_html_prefix, expected_html_contains[])
CASES = [
    (
        "# Heading\n\nParagraph.",
        "<h1>Heading</h1>",
        ["<p>Paragraph.</p>"],
    ),
    (
        "Paragraph with *emphasis* and **strong**.",
        "<p>Paragraph with ",
        ["<em>emphasis</em>", "<strong>strong</strong>"],
    ),
    (
        "> quote\n>\n> * list in quote",
        "<blockquote>",
        ["<ul>", "<li>list in quote</li>"],
    ),
    (
        "1. one\n2. two\n3. three",
        "<ol>",
        ["<li>one</li>", "<li>two</li>", "<li>three</li>"],
    ),
    (
        "```\ncode\n```",
        "<pre><code>code\n</code></pre>",
        [],
    ),
    (
        "[link](/url) and `code`",
        "<p>",
        ['<a href="/url">link</a>', "<code>code</code>"],
    ),
    (
        "---",
        "<hr />",
        [],
    ),
]

# Synthetic perf doc (nested lists, code, links, headings)
PERF_MD = (
    "# Title\n\n"
    + "\n".join(f"- item {i} with **bold** and *em* and [link](/u/{i})" for i in range(200))
    + "\n\n```\n" + "\n".join(f"line {i} of code" for i in range(400)) + "\n```"
)


def _load_program(program_path: str):
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    return program


def _build_crate(crate_dir: str, crate_name: str):
    # native
    code, out, err = _run(["cargo", "build", "--release"], cwd=crate_dir, timeout=600)
    if code != 0:
        raise RuntimeError(f"native build failed: {err.decode(errors='ignore')}")
    native_bin = str(Path(crate_dir) / "target" / "release" / crate_name)

    # wasm
    code_wasm, out_wasm, err_wasm = _run(
        ["cargo", "build", "--release", "--target", "wasm32-wasi"], cwd=crate_dir, timeout=900
    )
    wasm_ok = code_wasm == 0
    wasm_path = str(Path(crate_dir) / "target" / "wasm32-wasi" / "release" / f"{crate_name}.wasm") if wasm_ok else None
    if not wasm_ok:
        # Allow progress without WASM but cap final score later
        wasm_path = None

    return native_bin, wasm_path


def _run_binary(path: str, md: str, timeout_s=5):
    return _run([path], input_bytes=md.encode("utf-8"), timeout=timeout_s)


def _run_wasm(wasm_path: str, md: str, timeout_s=5):
    runner = _which("wasmtime") or _which("wasmer")
    if not runner:
        return 127, b"", b"no_wasm_runner"
    if Path(runner).name == "wasmtime":
        return _run([runner, wasm_path, "--dir=."], input_bytes=md.encode("utf-8"), timeout=timeout_s)
    else:
        # wasmer
        return _run([runner, "run", wasm_path], input_bytes=md.encode("utf-8"), timeout=timeout_s)


def _score_correctness(bin_path: str) -> float:
    passed = 0
    for md, html_prefix, contains in CASES:
        code, out, err = _run_binary(bin_path, md, timeout_s=3)
        if code != 0:
            continue
        html = out.decode("utf-8", errors="ignore")
        ok = html.strip().startswith(html_prefix) and all(x in html for x in contains)
        passed += 1 if ok else 0
    return passed / len(CASES)


def _score_perf(bin_path: str) -> float:
    # Measure time for multiple runs; map to [0,1] where 1.0 is very fast.
    trials = 5
    durations = []
    payload = PERF_MD
    for _ in range(trials):
        t0 = time.perf_counter()
        code, out, err = _run_binary(bin_path, payload, timeout_s=10)
        t1 = time.perf_counter()
        if code != 0:
            return 0.0
        durations.append(t1 - t0)
    # Robust central tendency
    durations.sort()
    med = durations[len(durations) // 2]

    # Scale: <= 40ms per run → 1.0; >= 300ms → 0.0
    if med <= 0.040:
        return 1.0
    if med >= 0.300:
        return 0.0
    # Linear interpolation
    return (0.300 - med) / (0.300 - 0.040)


def _score_size(wasm_path: str | None) -> float:
    if not wasm_path or not os.path.exists(wasm_path):
        return 0.0
    size = os.path.getsize(wasm_path)
    # 0.25MB or less → 1.0 ; 2.5MB or more → 0.0
    small = 256 * 1024
    large = 2_500 * 1024
    if size <= small:
        return 1.0
    if size >= large:
        return 0.0
    return (large - size) / (large - small)


def evaluate(program_path: str):
    """
    ADRS entrypoint. Returns a dict with `combined_score` and metrics.
    """
    try:
        program = _load_program(program_path)
        with tempfile.TemporaryDirectory() as tmp:
            mats = program.materialize_crate(tmp, crate_name="mdwasm")
            native_bin, wasm_path = _build_crate(mats["crate_dir"], mats["crate_name"])

            correctness = _score_correctness(native_bin)
            perf = _score_perf(native_bin)
            size = _score_size(wasm_path)

            # Encourage WASM success: if no wasm, cap at 0.6 total
            raw = 0.70 * correctness + 0.25 * perf + 0.05 * size
            if wasm_path is None:
                raw = min(raw, 0.60)

            result = {
                "combined_score": float(raw),
                "correctness": float(correctness),
                "perf": float(perf),
                "wasm_size_score": float(size),
                "wasm_bytes": int(os.path.getsize(wasm_path)) if wasm_path else 0,
                "built_wasm": 1.0 if wasm_path else 0.0,
            }
            return result
    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": str(e),
            "correctness": 0.0,
            "perf": 0.0,
            "wasm_size_score": 0.0,
            "built_wasm": 0.0,
            "wasm_bytes": 0,
        }


if __name__ == "__main__":
    # Quick self-check for local dev
    here = Path(__file__).parent
    prog = here / "initial_program.py"
    print(json.dumps(evaluate(str(prog)), indent=2))
