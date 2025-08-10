import os
import json
import glob
from pathlib import Path
from typing import List, Tuple

import torch
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

# Your inference helpers (already built in inference.py)
from inference import read_uploaded, run_check, SR

APP_TITLE = "AwaazShield — Voice Call Fraud & Deepfake Guard"
APP_TAGLINE = "Real-time voice spoof detection · speaker verification · passphrase check"

# Brand palette
PRIMARY = "#0f172a"   # slate-900
ACCENT  = "#22c55e"   # green-500
RED     = "#ef4444"
AMBER   = "#f59e0b"

# Utility: waveform
def plot_waveform(y: np.ndarray):
    fig = plt.figure(figsize=(7, 1.8))
    plt.plot(np.linspace(0, len(y)/SR, len(y)), y)
    plt.xlabel("Time (s)"); plt.ylabel("Amp"); plt.tight_layout()
    return fig

# UI cards
def card_html(label, value, sub=None, color=ACCENT):
    sub_html = f"<div style='opacity:.8;font-size:12px'>{sub}</div>" if sub else ""
    return f"""
    <div style="background:white;border-radius:16px;padding:16px 18px;box-shadow:0 8px 24px rgba(0,0,0,.08);border:1px solid rgba(0,0,0,.06)">
      <div style="font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:.08em">{label}</div>
      <div style="font-size:20px;font-weight:700;color:{color};margin-top:4px">{value}</div>
      {sub_html}
    </div>
    """

def decision_chip(label):
    color = {"Safe": ACCENT, "Caution": AMBER, "Danger": RED}.get(label, "#64748b")
    return f"<span style='background:{color};color:white;padding:6px 10px;border-radius:999px;font-weight:700'>{label}</span>"

# Example audio discovery
ROOT = Path(__file__).resolve().parents[1]  # repo root
SAMPLES_DIR = Path(__file__).resolve().parent / "sample_audio"

def list_samples() -> List[str]:
    if not SAMPLES_DIR.exists():
        return []
    files = sorted(glob.glob(str(SAMPLES_DIR / "*.wav"))) + sorted(glob.glob(str(SAMPLES_DIR / "*.mp3")))
    # Return relative paths (Gradio accepts file paths)
    return files

def make_example_pairs(files: List[str]) -> List[Tuple[str, str]]:
    """
    Heuristic: create a few reasonable pairs.
    - If we find files containing 'enroll' or 'owner', prefer as enrollment.
    - Otherwise, build the first 3 unique pairs.
    """
    if not files:
        return []
    enrolls = [f for f in files if any(k in Path(f).name.lower() for k in ["enroll","owner","legit","spk1"])]
    probes  = [f for f in files if f not in enrolls]
    pairs = []
    if enrolls and probes:
        for e in enrolls[:3]:
            for p in probes[:5]:
                if e != p:
                    pairs.append((e, p))
                    if len(pairs) >= 6: break
            if len(pairs) >= 6: break
    else:
        # fallback: first few combinations
        for i in range(min(3, len(files))):
            for j in range(i+1, min(i+1+3, len(files))):
                pairs.append((files[i], files[j]))
                if len(pairs) >= 6: break
            if len(pairs) >= 6: break
    return pairs

EXAMPLE_FILES = list_samples()
EXAMPLE_PAIRS = make_example_pairs(EXAMPLE_FILES)

# Latest report linking
def find_latest_report_dir() -> Path | None:
    reports_root = ROOT / "reports"
    if not reports_root.exists():
        return None
    candidates = [p for p in reports_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    # Sort by mtime descending
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def build_report_html() -> str:
    rep = find_latest_report_dir()
    if rep is None:
        return "<div>No report folder found. Run the report generator in the notebook to populate /reports.</div>"
    figs = list((rep / "figs").glob("*.png")) + list((rep / "figs").glob("*.pdf"))
    figs = sorted(figs)[:24]  # limit
    links = []
    for f in figs:
        rel = f.relative_to(ROOT)
        links.append(f"<li><a href='file/{rel.as_posix()}' target='_blank'>{rel.name}</a></li>")
    if not links:
        links_html = "<li>(no figures found)</li>"
    else:
        links_html = "\n".join(links)
    md = f"""
    <div style="line-height:1.5">
      <div><strong>Latest report:</strong> <code>{rep.name}</code></div>
      <div>Figures (click to open):</div>
      <ul>
        {links_html}
      </ul>
      <div style="opacity:.85">Full path: <code>{rep.as_posix()}</code></div>
    </div>
    """
    return md

# Core inference
def go(enroll, probe, expected_digits):
    if enroll is None or probe is None:
        return gr.update(value="Upload both enrollment and probe audio."), None, None, None, None, None

    # read audio from uploaded paths
    y_en = read_uploaded(enroll)
    y_pr = read_uploaded(probe)

    # run full check
    res = run_check(y_en, y_pr, expected_digits or "")

    # plots
    w1 = plot_waveform(y_en)
    w2 = plot_waveform(y_pr)

    # cards
    risk_pct = f"{res.risk*100:.1f}%"
    spk_str  = f"{res.speaker_score:.3f}"
    spf_str  = f"{res.spoof_prob:.3f}"
    phr_str  = f"{'PASS' if res.phrase_ok else 'FAIL'} ({res.digits_heard or '…'})"

    html = f"""
      <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px">
        {card_html("Decision", decision_chip(res.label), color='#111827')}
        {card_html("Overall Risk", risk_pct)}
        {card_html("Speaker Similarity (raw)", spk_str)}
        {card_html("Spoof Probability", spf_str)}
      </div>
      <div style="height:12px"></div>
      <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px">
        {card_html("Passphrase", phr_str, sub=f"Expected: {expected_digits or '—'}")}
        {card_html("Device", "GPU" if torch.cuda.is_available() else "CPU")}
        {card_html("Sample Rate", "16 kHz")}
        {card_html("Version", "MVP r1")}
      </div>
    """
    return html, w1, w2, f"{res.digits_heard or ''}", res.label, f"{res.risk:.3f}"

# Example loader 
def load_example(idx: int, expected_digits: str):
    if not EXAMPLE_PAIRS:
        return gr.update(), gr.update(), gr.update()
    e, p = EXAMPLE_PAIRS[int(idx) % len(EXAMPLE_PAIRS)]
    # return filepaths into the Audio components + digits field
    return e, p, expected_digits

# App 
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate")) as demo:
    # Make CPU free tier responsive
    demo.queue(concurrency_count=1, max_size=8)

    # Hero
    gr.HTML(f"""
    <div style="padding:28px 0 12px;background:{PRIMARY};color:white;border-radius:16px">
      <div style="max-width:980px;margin:0 auto;padding:0 16px">
        <div style="font-size:28px;font-weight:800;letter-spacing:-0.02em">{APP_TITLE}</div>
        <div style="opacity:.9;margin-top:6px">{APP_TAGLINE}</div>
      </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**1) Enrollment voice (the legitimate owner)**")
            enroll = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Enrollment Audio")
            gr.Markdown("**2) Probe voice (the incoming caller)**")
            probe  = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Probe Audio")
            expected = gr.Textbox(label="Passphrase digits (optional)", placeholder="e.g., 731")

            with gr.Row():
                btn = gr.Button("Run AwaazShield", variant="primary")
                clear = gr.Button("Clear")

            # Examples panel (if present)
            if EXAMPLE_PAIRS:
                example_names = [f"{Path(e).name}  →  {Path(p).name}" for e, p in EXAMPLE_PAIRS]
                gr.Markdown("**Try examples (instant demo):**")
                with gr.Row():
                    ex_idx = gr.Dropdown(choices=[str(i) for i in range(len(EXAMPLE_PAIRS))],
                                         value="0", label="Example Pair")
                    ex_digits = gr.Textbox(value="731", label="Digits for example", scale=1)
                    use_ex = gr.Button("Load Example", variant="secondary")

        with gr.Column(scale=1):
            out_html = gr.HTML()
            with gr.Accordion("Waveforms", open=False):
                wf1 = gr.Plot(label="Enrollment waveform")
                wf2 = gr.Plot(label="Probe waveform")
            with gr.Row():
                heard = gr.Textbox(label="Heard digits", interactive=False)
                label = gr.Textbox(label="Decision", interactive=False)
                risk  = gr.Textbox(label="Risk (0-1)", interactive=False)

    btn.click(fn=go, inputs=[enroll, probe, expected], outputs=[out_html, wf1, wf2, heard, label, risk])

    clear.click(lambda: (None, None, "", gr.update(value=""), gr.update(value=""), gr.update(value="")),
                inputs=[], outputs=[enroll, probe, expected, out_html, label, risk])

    if EXAMPLE_PAIRS:
        use_ex.click(fn=load_example, inputs=[ex_idx, ex_digits], outputs=[enroll, probe, expected])

    # Reports section (auto-links latest)
    gr.Markdown("### Reports")
    reports_panel = gr.HTML(build_report_html())

if __name__ == "__main__":
    # On Spaces, just running app/app.py is enough
    demo.launch()
