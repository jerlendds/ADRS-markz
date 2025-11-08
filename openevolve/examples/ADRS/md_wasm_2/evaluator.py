"""
Evaluator for the Rust Markdown→HTML renderer.

Contract:
- Import the candidate program file (path provided by OpenEvolve engine).
- Expect it to expose: write_rust_project(workdir), PROJECT_NAME
- Build native (for runtime) and wasm32 (to enforce wasm-safety).
- Run functional tests (subset of CommonMark) + throughput microbench.
- Return a combined_score in [0,1] with weights (correctness 0.7, speed 0.3).
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from statistics import median

class TimeoutError(Exception):
    pass

def _load_program(program_path):
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    return program

def _run(cmd, cwd=None, timeout=600):
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        raise TimeoutError(f"Command timed out: {' '.join(cmd)}\n{out}\n{err}")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nstdout:\n{out}\nstderr:\n{err}")
    return out, err

def _ensure_rust():
    # Soft check for cargo/rustc; raise helpful error if missing.
    for tool in ("cargo", "rustc"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"Missing required tool: {tool}. Install Rust (https://rustup.rs).")

def _build(program, workdir):
    proj = program.write_rust_project(workdir)
    proj_dir = proj["project_dir"]

    # Native build (for running)
    _run(["cargo", "build", "--release"], cwd=proj_dir, timeout=900)
    native_bin = os.path.join(
        proj_dir, "target", "release",
        proj["bin_name"] + (".exe" if os.name == "nt" else "")
    )

    # Try wasm32 build to enforce wasm-compat; do not execute it.
    wasm_ok = True
    try:
        _run(["cargo", "build", "--release", "--target", "wasm32-unknown-unknown"], cwd=proj_dir, timeout=900)
    except Exception:
        wasm_ok = False  # penalize but don’t hard-fail; evolution should learn to fix

    return native_bin, wasm_ok

# Small, spec-aligned sanity tests. Keep stable HTML to avoid style diffs.
TESTS = [
    # (markdown, expected_html)
    ("# Title\n", "<h1>Title</h1>\n"),
    ("## A *fast* parser\n\nText.", "<h2>A <em>fast</em> parser</h2>\n<p>Text.</p>\n"),
    ("> quote\n>\n> more", "<blockquote>\n<p>quote</p>\n<p>more</p>\n</blockquote>\n"),
    ("- a\n- b\n- **c**", "<ul>\n<li>a</li>\n<li>b</li>\n<li><strong>c</strong></li>\n</ul>\n"),
    ("Inline `code` and **bold**.", "<p>Inline <code>code</code> and <strong>bold</strong>.</p>\n"),
    ("```\n<raw>\n&keep\n```\n", "<pre><code><raw>\n&keep\n</code></pre>\n"),
    ("a [link](https://x.y)", "<p>a <a href=\"https://x.y\">link</a></p>\n"),
    ("Paragraph\ncontinues", "<p>Paragraph continues</p>\n"),
]

def _score_correctness(binary_path):
    ok = 0
    for md, expect in TESTS:
        proc = subprocess.Popen([binary_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate(md, timeout=10)
        if proc.returncode != 0:
            continue
        if out == expect:
            ok += 1
    return ok / len(TESTS)

def _bench_speed(binary_path, size_bytes=300_000, rounds=5):
    # Build a stress input: repeated mixed patterns.
    seed = (
        "# H1\n"
        "Some *emphasis* and **strong** with `code` and [link](https://e.x).\n\n"
        "> quote line\n- item 1\n- item 2\n\n"
        "```\ncode & <raw>\n```\n\n"
    )
    buf = (seed * (max(1, size_bytes // len(seed)))).encode("utf-8")
    times = []
    for _ in range(rounds):
        st = time.time()
        p = subprocess.Popen([binary_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Discard output; measure throughput of end-to-end render
        out, err = p.communicate(buf, timeout=30)
        if p.returncode != 0:
            # treat as slow
            times.append(30.0)
        else:
            times.append(time.time() - st)
    t = median(times)
    cps = len(buf) / max(1e-6, t)  # chars per second
    # Normalize speed to [0,1] vs a simple target. 3e6 cps ~ good; clamp.
    return max(0.0, min(1.0, cps / 3_000_000.0))

def evaluate(program_path):
    try:
        _ensure_rust()
        program = _load_program(program_path)
        with tempfile.TemporaryDirectory() as td:
            native_bin, wasm_ok = _build(program, td)
            correctness = _score_correctness(native_bin)
            speed = _bench_speed(native_bin)

            # combine: correctness (0.7), speed (0.25), wasm_ok (0.05)
            combined = 0.7 * correctness + 0.25 * speed + 0.05 * (1.0 if wasm_ok else 0.0)
            return {
                "combined_score": float(combined),
                "correctness": float(correctness),
                "speed_score": float(speed),
                "wasm_compiles": float(1.0 if wasm_ok else 0.0),
            }
    except Exception as e:
        print("Evaluation failed:", str(e))
        traceback.print_exc()
        return {
            "combined_score": 0.0,
            "correctness": 0.0,
            "speed_score": 0.0,
            "wasm_compiles": 0.0,
            "error": str(e),
        }

if __name__ == "__main__":
    # Local smoke test
    here = os.path.dirname(os.path.abspath(__file__))
    prog_path = os.path.join(here, "initial_program.py")
    print(evaluate(prog_path))
