"""
Initial OpenEvolve program for evolving a zero-dependency Rust Markdown->HTML
parser compiled to WebAssembly.

Contract:
- Exposes EVOLVE-BLOCK string "RUST_LIB_RS" that OpenEvolve mutates.
- Provides run_bench() that:
  1) materializes a Cargo project with the evolved Rust code
  2) builds both native (for fallback perf) and wasm32-wasi
  3) returns a dict with {html, elapsed_ms, wasm_size_bytes} for evaluator
"""

import os
import shutil
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path

# -------- EVOLVE-BLOCK-START
# The Rust library source to evolve. Zero external crates.
RUST_LIB_RS = """
#![forbid(unsafe_code)]
// Extremely small, zero-deps, partial CommonMark-ish baseline.
// Intent: give evolution something correct-but-minimal to improve.
// Design: single-pass line scanner, handles a safe subset:
// - headings (#, ##, ...) with space
// - paragraphs
// - emphasis/strong: *em*, **strong** (no nesting/edge cases)
// - inline code: `code`
// - code blocks: triple backticks fences
// - unordered lists: lines starting with '- '
// - links: [text](url) (single-line, no nested parens)
// This is NOT fully compliant; evaluator will push evolution toward spec.

pub fn render_html(input: &str) -> String {
    let mut out = String::with_capacity(input.len() * 2);
    let mut in_code_block = false;

    let mut lines = input.lines().peekable();
    while let Some(line) = lines.next() {
        let trimmed = line.trim_end();

        // fenced code block
        if trimmed.starts_with("```") {
            if in_code_block {
                out.push_str("</code></pre>\n");
                in_code_block = false;
            } else {
                out.push_str("<pre><code>");
                in_code_block = true;
            }
            continue;
        }
        if in_code_block {
            // Escape HTML entities minimally
            for ch in trimmed.chars() {
                match ch {
                    '&' => out.push_str("&amp;"),
                    '<' => out.push_str("&lt;"),
                    '>' => out.push_str("&gt;"),
                    _ => out.push(ch),
                }
            }
            out.push('\n');
            continue;
        }

        // heading
        if let Some((hashes, rest)) = split_heading(trimmed) {
            let level = hashes.len().min(6);
            let body = rest.trim_start();
            out.push('<');
            out.push('h');
            out.push(char::from(b'0' + level as u8));
            out.push('>');
            out.push_str(&inline(body));
            out.push_str("</h");
            out.push(char::from(b'0' + level as u8));
            out.push_str(">\n");
            continue;
        }

        // unordered list block detection
        if trimmed.starts_with("- ") {
            out.push_str("<ul>\n");
            // consume consecutive list items
            let mut current = Some(trimmed.to_string());
            while let Some(item_line) = current {
                let item = item_line.strip_prefix("- ").unwrap_or("").trim();
                out.push_str("<li>");
                out.push_str(&inline(item));
                out.push_str("</li>\n");

                // peek next
                if let Some(next) = lines.peek() {
                    if next.trim_start().starts_with("- ") {
                        current = Some(lines.next().unwrap().trim_end().to_string());
                        continue;
                    }
                }
                current = None;
            }
            out.push_str("</ul>\n");
            continue;
        }

        // blank line => paragraph separator
        if trimmed.is_empty() {
            out.push('\n');
            continue;
        }

        // paragraph
        out.push_str("<p>");
        out.push_str(&inline(trimmed));
        out.push_str("</p>\n");
    }

    out
}

// find leading '#' heading per CommonMark-style (requires space after hashes)
fn split_heading(line: &str) -> Option<(&str, &str)> {
    let bytes = line.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && bytes[i] == b'#' && i < 6 { i += 1; }
    if i == 0 { return None; }
    if i < bytes.len() && bytes[i] == b' ' {
        Some((&line[..i], &line[i+1..]))
    } else {
        None
    }
}

// very small inline processor: **strong**, *em*, `code`, [txt](url)
fn inline(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    let mut i = 0;
    let b = s.as_bytes();
    while i < b.len() {
        match b[i] {
            b'`' => {
                if let Some(j) = s[i+1..].find('`') {
                    out.push_str("<code>");
                    out.push_str(&escape_html(&s[i+1..i+1+j]));
                    out.push_str("</code>");
                    i += j + 2;
                } else {
                    out.push('`'); i += 1;
                }
            }
            b'*' => {
                // try **strong**
                if i + 1 < b.len() && b[i+1] == b'*' {
                    if let Some(j) = s[i+2..].find("**") {
                        out.push_str("<strong>");
                        out.push_str(&escape_html(&s[i+2..i+2+j]));
                        out.push_str("</strong>");
                        i += j + 4;
                    } else { out.push('*'); out.push('*'); i += 2; }
                } else {
                    if let Some(j) = s[i+1..].find('*') {
                        out.push_str("<em>");
                        out.push_str(&escape_html(&s[i+1..i+1+j]));
                        out.push_str("</em>");
                        i += j + 2;
                    } else { out.push('*'); i += 1; }
                }
            }
            b'[' => {
                if let Some(close) = s[i+1..].find(']') {
                    let text = &s[i+1..i+1+close];
                    let rest = &s[i+1+close+1..];
                    if rest.starts_with("(") {
                        if let Some(rp) = rest.find(')') {
                            let url = &rest[1..rp];
                            out.push_str("<a href=\"");
                            out.push_str(&escape_attr(url));
                            out.push_str("\">");
                            out.push_str(&escape_html(text));
                            out.push_str("</a>");
                            i += 1 + close + 1 + rp + 1;
                        } else { out.push('['); i += 1; }
                    } else { out.push('['); i += 1; }
                } else { out.push('['); i += 1; }
            }
            b'&' => { out.push_str("&amp;"); i += 1; }
            b'<' => { out.push_str("&lt;"); i += 1; }
            b'>' => { out.push_str("&gt;"); i += 1; }
            _ => { out.push(b[i] as char); i += 1; }
        }
    }
    out
}

fn escape_html(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            _ => o.push(ch),
        }
    }
    o
}

fn escape_attr(s: &str) -> String {
    // very small attribute escaper (quotes -> &quot;, and HTML entities)
    let mut o = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            _ => o.push(ch),
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn h1() { assert!(render_html("# Title").contains("<h1>Title</h1>")); }
    #[test]
    fn para() { assert_eq!(render_html("a"), "<p>a</p>\n"); }
    #[test]
    fn strong_em() { assert_eq!(render_html("**a** *b*"), "<p><strong>a</strong> <em>b</em></p>\n"); }
    #[test]
    fn code_span() { assert_eq!(render_html("`x<y`"), "<p><code>x&lt;y</code></p>\n"); }
}
"""
# -------- EVOLVE-BLOCK-END


CARGO_TOML = """
[package]
name = "oe_markdown"
version = "0.1.0"
edition = "2021"

[lib]
name = "oe_markdown"
path = "src/lib.rs"
crate-type = ["rlib", "cdylib"]

[[bin]]
name = "oe_md_cli"
path = "src/main.rs"
"""

MAIN_RS = """
use std::io::{self, Read};

fn main() {
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf).unwrap();
    let html = oe_markdown::render_html(&buf);
    println!("{html}");
}
"""


def _write_project(root: Path):
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "Cargo.toml").write_text(CARGO_TOML)
    (root / "src" / "lib.rs").write_text(RUST_LIB_RS)
    (root / "src" / "main.rs").write_text(MAIN_RS)


def _run(cmd, cwd: Path, timeout: int = 180):
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def run_bench():
    """
    Materialize, build (native + wasm32-wasi), run a quick sanity parse,
    and return metrics for evaluator:
      {
        "html": "<p>…</p>",
        "elapsed_ms": float,
        "wasm_size_bytes": int,
        "native_ok": 0/1,
        "wasm_ok": 0/1
      }
    """
    tmp = Path(tempfile.mkdtemp(prefix="oe_md_"))
    try:
        _write_project(tmp)

        # Build native first (often faster and gives nice errors).
        t0 = time.time()
        native = _run(["cargo", "build", "--release"], cwd=tmp, timeout=360)
        t1 = time.time()

        native_ok = int(native.returncode == 0)

        # Attempt wasm32-wasi build (the Rust code must be zero-deps so this should succeed if toolchain exists).
        wasm_ok = 0
        wasm_size = 0
        wasm_target_dir = tmp / "target" / "wasm32-wasi" / "release"
        wasm_bin = wasm_target_dir / "oe_md_cli.wasm"
        # Try to add the target if missing (ignore failure).
        _run(["rustup", "target", "add", "wasm32-wasi"], cwd=tmp, timeout=120)
        wasm = _run(["cargo", "build", "--release", "--target", "wasm32-wasi"], cwd=tmp, timeout=480)
        if wasm.returncode == 0 and wasm_bin.exists():
            wasm_ok = 1
            wasm_size = wasm_bin.stat().st_size

        # Quick execution test: prefer wasmtime on the WASI artifact; fallback to native bin.
        sample_md = "# Title\n\n- a\n- b\n\n`x<y` **z**"
        html = ""
        start = time.time()
        ran = False

        # Try wasmtime first (if present and wasm build ok).
        if wasm_ok:
            probe = shutil.which("wasmtime")
            if probe:
                p = _run(
                    [probe, str(wasm_bin)],
                    cwd=tmp,
                    timeout=5,
                )
                if p.returncode == 0:
                    # wasmtime’s CLI reads stdin by default if '-' is not required; if not, rerun with input
                    # Simpler: run with input through Popen for this path (keep it robust).
                    proc = subprocess.Popen([probe, str(wasm_bin), "-"], cwd=str(tmp),
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    out, err = proc.communicate(sample_md, timeout=5)
                    if proc.returncode == 0:
                        html = out
                        ran = True

        # Fallback to native CLI
        if not ran and native_ok:
            cli = tmp / "target" / "release" / "oe_md_cli"
            if os.name == "nt":
                cli = cli.with_suffix(".exe")
            if cli.exists():
                proc = subprocess.Popen([str(cli)], cwd=str(tmp),
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                out, err = proc.communicate(sample_md, timeout=3)
                if proc.returncode == 0:
                    html = out
                    ran = True

        end = time.time()

        elapsed_compile_ms = (t1 - t0) * 1000.0
        elapsed_exec_ms = max(0.0, (end - start) * 1000.0)

        return {
            "html": html.strip(),
            "elapsed_ms": float(elapsed_exec_ms),
            "compile_ms": float(elapsed_compile_ms),
            "wasm_size_bytes": int(wasm_size),
            "native_ok": int(native_ok),
            "wasm_ok": int(wasm_ok),
        }
    finally:
        # Keep artifacts for debugging? For evolution, we can clean up to
        # avoid disk bloat. Comment next line if you want to inspect builds.
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print(run_bench())
