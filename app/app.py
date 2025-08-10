import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from inference import read_uploaded, run_check, SR

APP_TITLE = "AwaazShield — Voice Call Fraud & Deepfake Guard"
APP_TAGLINE = "Real-time voice spoof detection · speaker verification · passphrase check"

# Simple brand color palette
PRIMARY = "#0f172a"   # slate-900
ACCENT  = "#22c55e"   # green-500
RED     = "#ef4444"
AMBER   = "#f59e0b"

def plot_waveform(y):
    fig = plt.figure(figsize=(7,1.8))
    plt.plot(np.linspace(0, len(y)/SR, len(y)), y)
    plt.xlabel("Time (s)"); plt.ylabel("Amp"); plt.tight_layout()
    return fig

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
    color = {"Safe":ACCENT, "Caution":AMBER, "Danger":RED}.get(label, "#64748b")
    return f"<span style='background:{color};color:white;padding:6px 10px;border-radius:999px;font-weight:700'>{label}</span>"

def go(enroll, probe, expected_digits):
    if enroll is None or probe is None:
        return gr.update(value="Upload both enrollment and probe audio."), None, None, None, None, None

    y_en = read_uploaded(enroll)
    y_pr = read_uploaded(probe)

    res = run_check(y_en, y_pr, expected_digits or "")

    w1 = plot_waveform(y_en)
    w2 = plot_waveform(y_pr)

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
        {card_html("Device", "GPU" if gr.cuda_available() else "CPU")}
        {card_html("Sample Rate", "16 kHz")}
        {card_html("Version", "MVP r1")}
      </div>
    """
    return html, w1, w2, f"{res.digits_heard or ''}", res.label, f"{res.risk:.3f}"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate")) as demo:
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

            btn = gr.Button("Run AwaazShield", variant="primary")
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

    gr.Markdown("### Reports")
    gr.Markdown(
        "Your full evaluation (ROC, PR, DET, calibration, confusion, threshold sweeps) "
        "is included in the app repository. If you pushed the `reports/` folder, "
        "judges can browse it in the repo or you can link selected PDFs in your deck."
    )

if __name__ == "__main__":
    demo.launch()
