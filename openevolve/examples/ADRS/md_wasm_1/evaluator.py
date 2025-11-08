"""
Evaluator for OpenEvolve Markdown->HTML (Rust/WASM, zero deps).

Interface: evaluate(program_path) -> dict with "combined_score" etc.
It imports the evolved program file (initial_program.py derivative),
calls run_bench() once, and also runs a deterministic correctness suite
of CommonMark-ish cases. Score blends correctness, speed, and wasm size.

Hard constraints (score=0 if violated):
- native_ok==1 (must compile natively as sanity fallback)
- wasm_ok==1 (must compile to wasm32-wasi)
- wasm_size_bytes <= 2_000_000  (keep it lean; adjusted easily)
- Must not emit unsafe code (we forbid in lib.rs baseline; quick grep here)
"""

import importlib.util
import math
import os
import re
import sys
import time
import traceback

STRICT_UNSAFE_RE = re.compile(r"\bunsafe\b")

# Minimal test corpus referencing CommonMark behaviors (subset).
# The idea is to be *spec-pulled* without importing big deps.
TESTS = [
    # headings
    ("# H", "<h1>H</h1>"),
    ("## H2", "<h2>H2</h2>"),
    ("####### not a heading", "<p>####### not a heading</p>"),  # 7 #'s -> still heading level 6 by many impls; baseline requires space, no 6+ cap here -> accept paragraph baseline
    # paragraphs & HTML escaping
    ("a < b & c", "<p>a &lt; b &amp; c</p>"),
    # emphasis / strong
    ("*em* **strong**", "<p><em>em</em> <strong>strong</strong></p>"),
    # code span escapes
    ("`x<y`", "<p><code>x&lt;y</code></p>"),
    # lists
    ("- a\n- b", "<ul>\n<li>a</li>\n<li>b</li>\n</ul>"),
    # link
    ("[x](https://e.x)", '<p><a href="https://e.x">x</a></p>'),
    # fenced code block
    ("```\n<>&\n```\n", "<pre><code>&lt;&gt;&amp;\n</code></pre>"),
]

def normalize_html(h: str) -> str:
    return re.sub(r"\s+", " ", h.strip())

def score_correctness(html_func):
    ok = 0
    for md, want in TESTS:
        got = html_func(md)
        if normalize_html(want) in normalize_html(got):
            ok += 1
    return ok / len(TESTS)


def evaluate(program_path: str):
    try:
        # Import the program under evaluation
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        if not hasattr(program, "run_bench"):
            return {
                "combined_score": 0.0,
                "runs_successfully": 0.0,
                "error": "Missing run_bench()",
            }

        # Quick grep to guard unsafe (best-effort; the lib forbids it at compile too).
        with open(program_path, "r", encoding="utf-8") as f:
            src = f.read()
        if STRICT_UNSAFE_RE.search(src):
            return {
                "combined_score": 0.0,
                "runs_successfully": 0.0,
                "error": "unsafe not allowed",
            }

        t0 = time.time()
        result = program.run_bench()
        t1 = time.time()
        wall_ms = (t1 - t0) * 1000.0

        # Hard constraints
        if int(result.get("native_ok", 0)) != 1 or int(result.get("wasm_ok", 0)) != 1:
            return {
                "combined_score": 0.0,
                "runs_successfully": 0.0,
                "error": "Build failed (native/wasm).",
            }

        wasm_size = int(result.get("wasm_size_bytes", 0))
        if wasm_size <= 0 or wasm_size > 2_000_000:
            return {
                "combined_score": 0.0,
                "runs_successfully": 0.0,
                "error": f"Invalid wasm size: {wasm_size}",
            }

        # Correctness on our subset
        html_str = result.get("html", "")
        # Provide a function for correctness scoring by reusing the binary path via run_bench?
        # We only have the one sample output; so we also call a tiny, embedded interpreter:
        # The easiest is to re-import program and reconstruct the Rust binary per run_bench.
        # For speed, just use the sample; and measure a correctness proxy by re-running run_bench on each test? Too slow.
        # Instead, accept a cheap proxy: baseline parser patterns in HTML. For selection pressure we still want real tests.
        # We'll just use program.run_bench()’s compiled native binary via a helper if provided.
        # If not provided, fall back to a Python-only proxy using the single output (weak). To keep deterministic, we
        # ship a tiny shim: program_html(md) -> runs the same compiled binary via stdin. If it's absent, we skip.

        if hasattr(program, "program_html"):
            html_fn = program.program_html  # (md: str) -> str
        else:
            # As a minimal fallback (won't drive great evolution but prevents zero):
            def html_fn(md: str) -> str:
                # one-shot regexy placeholder; will be dominated by build+size score
                return html_str if md.strip() else ""
        correctness = score_correctness(html_fn)

        # Speed: prefer the execution time returned by run_bench (elapsed_ms)
        elapsed_ms = float(result.get("elapsed_ms", wall_ms))
        compile_ms = float(result.get("compile_ms", 0.0))

        # Composite score:
        #   correctness: 0.70
        #   speed (lower better): 0.20 -> scale via 1 / (1 + elapsed_ms/2)
        #   wasm size (smaller better): 0.10 -> scale vs 1MB reference
        sp = 1.0 / (1.0 + max(0.0, elapsed_ms) / 2.0)  # ~2ms -> 0.33 penalty; favors sub-ms
        size_ref = 1_000_000.0
        sz = 1.0 / (1.0 + (max(1.0, wasm_size) / size_ref))

        combined = 0.70 * correctness + 0.20 * sp + 0.10 * sz

        return {
            "combined_score": float(combined),
            "runs_successfully": 1.0,
            "correctness": float(correctness),
            "speed_score": float(sp),
            "size_score": float(sz),
            "elapsed_ms": float(elapsed_ms),
            "compile_ms": float(compile_ms),
            "wasm_size_bytes": int(wasm_size),
        }

    except Exception as e:
        print("Evaluation failed:", str(e))
        print(traceback.format_exc())
        return {
            "combined_score": 0.0,
            "runs_successfully": 0.0,
            "error": str(e),
        }
