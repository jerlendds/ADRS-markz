"""
OpenEvolve (ADRS) initial program for evolving a Rust→WASM Markdown renderer.

This module exposes a single evolvable block: `rust_main_code`.
The evaluator will materialize a Cargo project using this string, build it
for native and wasm32-wasi, then exercise correctness & speed.

Binary contract (both native and WASI binary):
- Read UTF-8 Markdown from STDIN
- Write UTF-8 HTML to STDOUT
- Exit 0 on success; nonzero on error

Rust constraints:
- Must compile on stable Rust.
- CommonMark-compliant behavior (no extensions) by default.
- Prefer speed and small code size.

You can import this module and call `materialize_crate(out_dir)` to write files.
"""

import os
from pathlib import Path
from textwrap import dedent

# EVOLVE-BLOCK-START
# The entire Rust binary source is defined here and written to src/main.rs.
# OpenEvolve will mutate this string during evolution.
rust_main_code = """
use std::io::{self, Read, Write};

/// Renderer interface:
/// - read entire stdin as UTF-8 Markdown
/// - render to HTML (CommonMark)
/// - write to stdout
/// - exit 0 on success
///
/// Baseline uses `comrak` (CommonMark) with extensions disabled for spec compliance.
/// Tuning levers for evolution:
/// - Parser/arena allocation strategies
/// - Options affecting speed vs. correctness (keep CommonMark on!)
/// - String handling (minimize allocations/copies)
/// - Optional streaming partitioning while preserving correctness
/// - Feature flags to reduce code size
///
/// NOTE: Do not print logs to stdout; use stderr if needed.
fn main() {
    // Read all stdin
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();

    // Prepare options (CommonMark-only)
    // comrak's default is CommonMark + some safe defaults; we explicitly disable extensions.
    let mut options = comrak::ComrakOptions::default();
    options.extension.autolink = false;
    options.extension.table = false;
    options.extension.tasklist = false;
    options.extension.strikethrough = false;
    options.extension.tagfilter = false;
    options.extension.footnotes = false;
    options.extension.description_lists = false;
    options.parse.smart = false;
    options.render.hardbreaks = false;
    options.render.github_pre_lang = false;
    options.render.escape = true;

    // Use a single typed-arena per render to keep allocations fast & localized.
    // For performance tuning, an evolutionary step could reuse arenas across calls
    // in a worker process; the CLI contract is single-shot, so this is fine.
    let arena = comrak::Arena::new();

    // Parse → format
    let root = comrak::parse_document(&arena, &input, &options);
    let mut out = Vec::with_capacity(input.len().saturating_mul(2)); // heuristic
    comrak::format_html(root, &options, &mut out).unwrap();

    // Write to stdout
    io::stdout().write_all(&out).unwrap();
}
"""
# EVOLVE-BLOCK-END


def cargo_toml(crate_name: str = "mdwasm") -> str:
    # Keep dependency small and pinned; comrak is the CommonMark reference-grade crate.
    return dedent(f"""\
        [package]
        name = "{crate_name}"
        version = "0.1.0"
        edition = "2021"
        publish = false

        [dependencies]
        comrak = {{ version = "0.24", default-features = false, features = ["simd"] }}

        [profile.release]
        codegen-units = 1
        lto = true
        opt-level = "z"  # prefer small; evolution may change this
        panic = "abort"
        strip = "symbols"

        # Build a single binary; evaluator will compile both native & wasm32-wasi.
        [[bin]]
        name = "{crate_name}"
        path = "src/main.rs"
    """)


def materialize_crate(out_dir: str, crate_name: str = "mdwasm") -> dict:
    """
    Writes a minimal Cargo project using the evolvable Rust source.

    Returns a dict with useful paths.
    """
    root = Path(out_dir)
    (root / "src").mkdir(parents=True, exist_ok=True)

    (root / "Cargo.toml").write_text(cargo_toml(crate_name), encoding="utf-8")
    (root / "src" / "main.rs").write_text(rust_main_code, encoding="utf-8")

    return {
        "crate_dir": str(root),
        "cargo_toml": str(root / "Cargo.toml"),
        "main_rs": str(root / "src" / "main.rs"),
        "crate_name": crate_name,
    }
