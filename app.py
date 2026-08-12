"""Gradio UI for the Grounded Research Assistant.

Bring your own Anthropic key. The key is used only for your request and is never stored or logged.
Run with:  python app.py
"""
import os
import gradio as gr

INTRO = """# Grounded Research Assistant
Ask a science question. The assistant retrieves **real peer-reviewed research** (OpenAlex + arXiv),
answers **only from those sources, with citations**, using Claude, then **verifies its own answer**
and **refuses** when the sources do not support one.

Your Anthropic key is used only for this request and is never stored.
"""


def run(api_key: str, question: str):
    if not api_key or not api_key.startswith("sk-ant-"):
        return "Please enter a valid Anthropic API key (it starts with `sk-ant-`).", "", ""
    if not question or not question.strip():
        return "Please enter a question.", "", ""
    os.environ["ANTHROPIC_API_KEY"] = api_key.strip()
    try:
        from src import agent   # imported after the key is set
        result = agent.answer(question.strip())
    except Exception as exc:                       # surface any error to the user, never crash the UI
        return f"Error: {exc}", "", ""

    papers = result.get("papers", [])
    sources = "\n".join(
        f"[{i}] {p['title']} ({p.get('year')}, {p.get('venue')})   {p.get('doi') or ''}"
        for i, p in enumerate(papers, 1)
    ) or "No sources found."
    return result.get("answer", ""), result.get("grounding", ""), sources


with gr.Blocks(title="Grounded Research Assistant") as demo:
    gr.Markdown(INTRO)
    key = gr.Textbox(label="Anthropic API key", type="password", placeholder="sk-ant-...")
    question = gr.Textbox(label="Your question",
                          placeholder="What limits the cycle life of lithium-ion battery cathodes?")
    ask = gr.Button("Ask", variant="primary")
    answer = gr.Markdown(label="Answer (with citations)")
    grounding = gr.Textbox(label="Grounding check (the assistant checking itself)", interactive=False)
    sources = gr.Textbox(label="Sources used", interactive=False, lines=6)
    ask.click(run, inputs=[key, question], outputs=[answer, grounding, sources])


if __name__ == "__main__":
    demo.launch()
