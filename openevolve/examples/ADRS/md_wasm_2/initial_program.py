"""
Initial program for evolving a zero-dependency Rust Markdown -> HTML renderer
that must compile for wasm32-unknown-unknown and run as a native CLI for eval.

The evaluator will import this module and expect:
- write_rust_project(workdir: str) -> dict with paths
- PROJECT_NAME (str)

Only the code in the EVOLVE-BLOCK should be modified by OpenEvolve.
Everything else is harness/IO and should stay stable.
"""

import os
from textwrap import dedent

PROJECT_NAME = "md2html_zero_dep"

# ----------------------- EVOLVE TARGET (Rust) -----------------------
# Edits must stay within EVOLVE-BLOCK-START/END. Keep it std-only.
RUST_MAIN_CODE = """
//! Zero-dependency Markdown -> HTML renderer.
//! Must compile for native and wasm32-unknown-unknown.
#![forbid(unsafe_code)]

use std::io::{self, Read};

fn main() {
    // Read all stdin to a string, render, print to stdout
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let html = render_markdown(&input);
    println!("{}", html);
}

/// Public entry used by tests; keep stable signature for wasm.
/// Deterministic, no allocs beyond std containers, no threads.
pub fn render_markdown(src: &str) -> String {
    // EVOLVE-BLOCK-START
    // Baseline: a tiny, spec-inspired but incomplete CommonMark renderer.
    // Constraints: single pass with minimal state; good asymptotics, no deps.
    // Current coverage: headings (#..), paragraphs, emphasis (*, **),
    // code spans (`), fenced code blocks (```), blockquotes (>), ulist (- ),
    // links [text](url). Tight/loose list heuristics. Normalizes \r\n.
    // Known non-conformances: nested lists, intraword emphasis rules, escapes.
    // Optimize correctness & speed; must remain wasm-safe & no dependencies.

    enum Block {
        Para(Vec<String>),
        Heading(usize, String),
        CodeFence(String), // raw literal, no HTML escaping needed here
        Quote(Vec<String>),
        UList(Vec<String>), // items already in HTML li form
        Html(String),
    }

    fn html_escape(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for ch in s.chars() {
            match ch {
                '&' => out.push_str("&amp;"),
                '<' => out.push_str("&lt;"),
                '>' => out.push_str("&gt;"),
                '"' => out.push_str("&quot;"),
                '\'' => out.push_str("&#39;"),
                _ => out.push(ch),
            }
        }
        out
    }

    fn render_inlines(s: &str) -> String {
        // Very small inline parser with a single scan.
        let mut out = String::with_capacity(s.len() + s.len() / 8);
        let mut i = 0;
        let bytes = s.as_bytes();
        let n = bytes.len();

        let mut strong = false;
        let mut emph = false;
        let mut code = false;

        while i < n {
            // code spans
            if !code && bytes[i] == b'`' {
                code = true;
                out.push_str("<code>");
                i += 1;
                continue;
            }
            if code && bytes[i] == b'`' {
                code = false;
                out.push_str("</code>");
                i += 1;
                continue;
            }
            if code {
                // raw, no escaping (code span renders literal)
                out.push(bytes[i] as char);
                i += 1;
                continue;
            }

            // strong vs emphasis (greedy ** over *)
            if i + 1 < n && bytes[i] == b'*' && bytes[i + 1] == b'*' {
                if strong {
                    out.push_str("</strong>");
                } else {
                    out.push_str("<strong>");
                }
                strong = !strong;
                i += 2;
                continue;
            }
            if bytes[i] == b'*' {
                if emph {
                    out.push_str("</em>");
                } else {
                    out.push_str("<em>");
                }
                emph = !emph;
                i += 1;
                continue;
            }

            // link: [text](url) (no nesting, simple)
            if bytes[i] == b'[' {
                // find closing ] and following (..)
                let start_text = i + 1;
                if let Some(close_br) = s[start_text..].find(']') {
                    let end_text = start_text + close_br;
                    if end_text + 1 < n && bytes[end_text + 1] == b'(' {
                        if let Some(close_par) = s[end_text + 2..].find(')') {
                            let url_end = end_text + 2 + close_par;
                            let text = &s[start_text..end_text];
                            let url = &s[end_text + 2..url_end];
                            out.push_str("<a href=\"");
                            out.push_str(&html_escape(url));
                            out.push_str("\">");
                            out.push_str(&html_escape(text));
                            out.push_str("</a>");
                            i = url_end + 1;
                            continue;
                        }
                    }
                }
            }

            // default: escape
            out.push_str(&html_escape(&(bytes[i] as char).to_string()));
            i += 1;
        }

        // close any dangling marks (permissive)
        if code { out.push_str("</code>"); }
        if strong { out.push_str("</strong>"); }
        if emph { out.push_str("</em>"); }
        out
    }

    // Normalize newlines
    let src = src.replace("\r\n", "\n").replace('\r', "\n");
    let mut blocks: Vec<Block> = Vec::new();

    // Block pass
    let mut lines = src.lines().peekable();
    'outer: while let Some(line) = lines.next() {
        let l = line.trim_end();

        // fenced code block
        if l.starts_with("```") {
            let mut buf = String::new();
            while let Some(inner) = lines.next() {
                if inner.trim_end().starts_with("```") {
                    blocks.push(Block::CodeFence(buf));
                    continue 'outer;
                } else {
                    buf.push_str(inner);
                    buf.push('\n');
                }
            }
            // unclosed fence: treat as code anyway
            blocks.push(Block::CodeFence(buf));
            continue;
        }

        // heading ATX (# ..)
        {
            let mut hashes = 0usize;
            for ch in l.chars() {
                if ch == '#' { hashes += 1; } else { break; }
            }
            if hashes > 0 && hashes <= 6 {
                let rest = l[hashes..].trim_start();
                if !rest.is_empty() {
                    blocks.push(Block::Heading(hashes, rest.to_string()));
                    continue;
                }
            }
        }

        // blockquote
        if l.starts_with('>') {
            let mut buf = String::new();
            let mut first = true;
            let mut current = l;
            loop {
                let trimmed = current.trim_start_matches('>').trim_start();
                if !first { buf.push('\n'); }
                buf.push_str(trimmed);
                first = false;
                match lines.peek() {
                    Some(nxt) if nxt.trim_start().starts_with('>') => {
                        current = lines.next().unwrap();
                    }
                    _ => break,
                }
            }
            blocks.push(Block::Quote(vec![buf]));
            continue;
        }

        // unordered list (- space)
        if l.starts_with("- ") {
            let mut items: Vec<String> = Vec::new();
            let mut cur = l;
            loop {
                let content = cur[2..].trim_end();
                items.push(format!("<li>{}</li>", render_inlines(content)));
                match lines.peek() {
                    Some(nxt) if nxt.trim_start().starts_with("- ") => {
                        cur = lines.next().unwrap().trim_end();
                    }
                    _ => break,
                }
            }
            blocks.push(Block::UList(items));
            continue;
        }

        // blank -> paragraph boundary
        if l.trim().is_empty() {
            continue;
        }

        // paragraph (consume following tight lines)
        let mut para: Vec<String> = vec![l.to_string()];
        while let Some(peek) = lines.peek() {
            let pl = peek.trim_end();
            if pl.is_empty() || pl.starts_with('#') || pl.starts_with('>') or pl.starts_with("- ") or pl.starts_with("```") {
                break;
            }
            para.push(lines.next().unwrap().trim_end().to_string());
        }
        blocks.push(Block::Para(para));
    }

    // Emit HTML
    let mut out = String::with_capacity(src.len() + src.len() / 4);
    for b in blocks {
        match b {
            Block::Heading(level, text) => {
                out.push_str(&format!("<h{l}>{}</h{l}>\n", render_inlines(&text), l=level));
            }
            Block::Para(lines) => {
                let joined = lines.join(" ");
                out.push_str("<p>");
                out.push_str(&render_inlines(&joined));
                out.push_str("</p>\n");
            }
            Block::CodeFence(raw) => {
                // escape content
                out.push_str("<pre><code>");
                out.push_str(&html_escape(&raw));
                out.push_str("</code></pre>\n");
            }
            Block::Quote(chunks) => {
                out.push_str("<blockquote>\n");
                for chunk in chunks {
                    out.push_str("<p>");
                    out.push_str(&render_inlines(&chunk));
                    out.push_str("</p>\n");
                }
                out.push_str("</blockquote>\n");
            }
            Block::UList(items) => {
                out.push_str("<ul>\n");
                for it in items {
                    out.push_str(&it);
                    out.push('\n');
                }
                out.push_str("</ul>\n");
            }
            Block::Html(s) => { out.push_str(&s); out.push('\n'); }
        }
    }

    out
    // EVOLVE-BLOCK-END
}
"""


CARGO_TOML = lambda name: dedent(f"""\
    [package]
    name = "{name}"
    version = "0.1.0"
    edition = "2021"

    [profile.release]
    lto = "thin"
    codegen-units = 1
    opt-level = 3
    panic = "abort"
    debug = false
    incremental = false

    [dependencies]
    # Intentionally empty: zero-dependency requirement.

    [lib]
    name = "{name}"
    path = "src/lib.rs"
    crate-type = ["cdylib", "rlib"]

    [[bin]]
    name = "{name}"
    path = "src/main.rs"
    """)

LIB_RS = """
// keep lib exporting render_markdown for wasm linking/tests
#![forbid(unsafe_code)]

pub use crate::mainmod::render_markdown;

mod mainmod;
"""


def write_rust_project(workdir: str):
    """
    Materialize a Cargo project in `workdir/{PROJECT_NAME}` using RUST_MAIN_CODE.
    Returns paths dict used by the evaluator.
    """
    proj_dir = os.path.join(workdir, PROJECT_NAME)
    src_dir = os.path.join(proj_dir, "src")
    os.makedirs(src_dir, exist_ok=True)

    with open(os.path.join(proj_dir, "Cargo.toml"), "w") as f:
        f.write(CARGO_TOML(PROJECT_NAME))

    # split into lib + main module so wasm builds can link function
    with open(os.path.join(src_dir, "main.rs"), "w") as f:
        f.write('mod mainmod;\nfn main(){ mainmod::main(); }\n')

    with open(os.path.join(src_dir, "mainmod.rs"), "w") as f:
        f.write(RUST_MAIN_CODE)

    with open(os.path.join(src_dir, "lib.rs"), "w") as f:
        f.write(LIB_RS)

    return {
        "project_dir": proj_dir,
        "cargo_toml": os.path.join(proj_dir, "Cargo.toml"),
        "src_main": os.path.join(src_dir, "main.rs"),
        "src_mainmod": os.path.join(src_dir, "mainmod.rs"),
        "src_lib": os.path.join(src_dir, "lib.rs"),
        "bin_name": PROJECT_NAME,
    }
