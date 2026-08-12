"""Command-line interface.

    python -m src.cli "what limits lithium-ion cathode cycle life?"    # full grounded answer
    python -m src.cli --search-only "solid state electrolyte"          # just the sources (no key)
    python -m src.cli --source arxiv --search-only "graph neural net"  # preprints instead

--search-only needs no Anthropic key: it is the free way to verify the trusted-content layer.
"""
import argparse

from . import content, config


def _print_papers(papers: list) -> None:
    if not papers:
        print("No sources found.")
        return
    for i, p in enumerate(papers, 1):
        print(f"[{i}] {p['title']} ({p.get('year')}, {p.get('venue')})")
        print(f"     {p.get('doi') or ''}   | {p.get('source')}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Grounded Research Assistant")
    ap.add_argument("question", nargs="+", help="the question (or search query)")
    ap.add_argument("--search-only", action="store_true", help="only retrieve sources (no LLM, no key)")
    ap.add_argument("--source", choices=["openalex", "arxiv"], default="openalex",
                    help="which source to use with --search-only")
    args = ap.parse_args(argv)
    q = " ".join(args.question)

    if args.search_only:
        papers = content.search_arxiv(q) if args.source == "arxiv" else content.search_openalex(q)
        _print_papers(papers)
        return

    config.require_anthropic_key()
    from . import agent   # imported lazily so --search-only needs no LLM libraries or key
    result = agent.answer(q)
    print("\nANSWER\n------\n", result["answer"])
    print("\nGROUNDING CHECK\n--------------\n", result["grounding"])
    print("\nSOURCES\n-------")
    _print_papers(result["papers"])


if __name__ == "__main__":
    main()
