"""The grounded assistant: a LangGraph pipeline.

    retrieve  ->  synthesize  ->  verify

- retrieve   : pull real papers (trusted-content layer).
- synthesize : Claude drafts an answer using ONLY those papers, citing each claim.
- verify     : Claude re-reads its own answer against the sources and flags anything unsupported.

The responsible-AI rules live in the prompts: use only the sources, cite every claim, refuse when
the sources do not cover the question, and never fabricate.
"""
from typing import TypedDict, List

from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from . import config, content

log = config.get_logger("agent")

SYNTH_RULES = (
    "You are a scientific research assistant for professional users. "
    "Answer the question using ONLY the numbered sources provided. "
    "Cite every claim with its source number like [1], [2]. "
    "If the sources do not cover part of the question, say so plainly. "
    "Never invent facts, numbers, or citations. Be thorough and well-organised, "
    "drawing out the specific mechanisms and findings the sources actually contain."
)

VERIFY_RULES = (
    "You are a grounding checker. Given an ANSWER and its SOURCES, list any statements in the "
    "answer that are NOT supported by the sources. If every statement is supported, reply exactly: "
    "All claims grounded."
)

REFUSAL = ("I could not find supporting sources for this question, so I will not answer it. "
           "(No fabrication.)")


class State(TypedDict):
    question: str
    papers: List[dict]
    context: str
    answer: str
    grounding: str


def _llm() -> ChatAnthropic:
    # A fresh client each call so it always reads the current ANTHROPIC_API_KEY (the app sets it
    # per request). Newer Claude models manage sampling themselves, so no temperature is passed.
    return ChatAnthropic(model=config.MODEL)


def _format_context(papers: list) -> str:
    return "\n\n".join(
        f"[{i}] {p['title']} ({p.get('year')}, {p.get('venue')}). "
        f"{p.get('abstract', '')[: config.CONTEXT_ABSTRACT_CHARS]}"
        for i, p in enumerate(papers, 1)
    )


def retrieve_node(state: State) -> dict:
    return {"papers": content.retrieve(state["question"])}


def synthesize_node(state: State) -> dict:
    papers = state["papers"]
    if not papers:                                  # responsible AI: no sources -> no answer
        return {"answer": REFUSAL, "context": ""}
    ctx = _format_context(papers)
    resp = _llm().invoke([
        SystemMessage(content=SYNTH_RULES),
        HumanMessage(content=f"Question: {state['question']}\n\nSources:\n{ctx}"),
    ])
    return {"answer": resp.content, "context": ctx}


def verify_node(state: State) -> dict:
    if not state.get("context"):
        return {"grounding": "No sources were retrieved; the assistant correctly refused to answer."}
    resp = _llm().invoke([
        SystemMessage(content=VERIFY_RULES),
        HumanMessage(content=f"ANSWER:\n{state['answer']}\n\nSOURCES:\n{state['context']}"),
    ])
    return {"grounding": resp.content}


def build_assistant():
    """Compile the retrieve -> synthesize -> verify graph."""
    g = StateGraph(State)
    g.add_node("retrieve", retrieve_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("verify", verify_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "synthesize")
    g.add_edge("synthesize", "verify")
    g.add_edge("verify", END)
    return g.compile()


def answer(question: str) -> dict:
    """Run the full pipeline. Returns {question, papers, context, answer, grounding}."""
    config.require_anthropic_key()
    return build_assistant().invoke({"question": question})
