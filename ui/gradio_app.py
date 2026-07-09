"""Gradio demo — abstention is the star.

A live threshold slider flips questions between answer<->abstain in real time
(demoing the risk-coverage trade-off — the interview hook). A `--demo` flag
serves from cached generations so a free HF Space works with the GPU off.

Run:  python ui/gradio_app.py            # live (needs LLM_BASE_URL)
      python ui/gradio_app.py --demo     # cached generations only
"""
from __future__ import annotations

import argparse

from trustrag.config import load_config

FALSE_PREMISE_EXAMPLE = "What year did Albert Einstein win the Nobel Prize for the theory of relativity?"
NORMAL_EXAMPLE = "What is the boiling point of water at sea level in Celsius?"


def build_demo(demo_mode: bool):
    import gradio as gr

    cfg = load_config()
    from trustrag.pipeline import build_pipeline
    # NLI must ALWAYS be enabled when the gate is active: the gate's scaler and
    # coefficients were fitted on a feature set that includes nli_entail_max and
    # nli_contradict_max. Feeding zero NLI features violates the feature-parity
    # contract and biases the gate toward abstaining. In --demo mode, rely on
    # cached generations (so it's fast) but still run NLI (it's a small CPU model).
    pipe = build_pipeline(cfg, with_gate=True, with_nli=True)

    def run(query, documents, threshold):
        pages = [{"page_url": "", "page_name": f"doc{i}", "text": d}
                 for i, d in enumerate((documents or "").split("\n---\n")) if d.strip()]
        ans = pipe.answer(query, pages, threshold_override=float(threshold))
        verdict = "🟢 ANSWERED" if not ans.abstained else "🔴 ABSTAINED (I don't know)"
        conf_bar = f"confidence = {ans.confidence:.3f}   |   threshold τ = {threshold:.2f}"
        cites = "\n".join(f"- {c}" for c in ans.citations) or "(none)"
        return verdict, conf_bar, ans.text, cites

    with gr.Blocks(title="TrustRAG") as demo:
        gr.Markdown("# TrustRAG — Grounded QA with a Calibrated Abstention Gate\n"
                    "Slide the threshold τ to watch the model flip between **answering** "
                    "and **abstaining** — the risk-coverage trade-off, live.")
        query = gr.Textbox(label="Question", value=NORMAL_EXAMPLE)
        documents = gr.Textbox(label="Evidence pages (separate with a line: ---)", lines=6)
        threshold = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Abstention threshold τ")
        btn = gr.Button("Answer", variant="primary")
        verdict = gr.Markdown()
        conf = gr.Markdown()
        answer_box = gr.Textbox(label="Answer")
        cites = gr.Markdown(label="Citations")
        btn.click(run, [query, documents, threshold], [verdict, conf, answer_box, cites])
        gr.Examples([[NORMAL_EXAMPLE], [FALSE_PREMISE_EXAMPLE]], inputs=[query])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="serve from cached generations only")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build_demo(args.demo).launch(server_port=args.port)
