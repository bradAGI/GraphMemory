"""End-to-end test: ingest aimav4.txt using parallel LLM extraction via DSPy."""

import sys
import os
import re
import time
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dspy
from graphmemory import GraphMemory, MergeStrategy
from graphmemory.extraction import extract_and_merge_parallel

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Configure DSPy with gpt-5-nano (10k RPM, 10M TPM) ---
lm = dspy.LM("openai/gpt-5-nano")
dspy.configure(lm=lm)

# With 10k RPM we can safely run 50+ concurrent requests
MAX_WORKERS = 50


def chunk_text(text: str, max_chars: int = 4000) -> list[str]:
    """Split text into paragraph-aware chunks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = []
    current_len = 0
    for p in paragraphs:
        if current_len + len(p) > max_chars and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(p)
        current_len += len(p)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def on_progress(phase, done, total):
    bar_len = 30
    filled = int(bar_len * done / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  {phase:5s} [{bar}] {done}/{total}", end="", flush=True)
    if done == total:
        print()


def main():
    input_path = os.path.join(os.path.dirname(__file__), "..", "input", "aimav4.txt")
    with open(input_path) as f:
        text = f.read(200_000)

    text = re.sub(r"<!--.*?-->", "", text)
    chunks = chunk_text(text, max_chars=4000)

    print("=" * 60)
    print("GraphMemory — Parallel LLM Extraction")
    print("=" * 60)
    print(f"Source: aimav4.txt ({len(text):,} chars)")
    print(f"Chunks: {len(chunks)} x ~4k chars")
    print(f"Workers: {MAX_WORKERS} concurrent LLM calls")
    print(f"LLM: gpt-5-nano via DSPy")

    db = GraphMemory(database=":memory:", vector_length=3)

    print(f"\n--- Phase 1: Node extraction (parallel) ---")
    print(f"--- Phase 2: Edge extraction (parallel) ---")
    t0 = time.time()

    node_results, edge_results = extract_and_merge_parallel(
        db,
        chunks,
        match_keys=["name"],
        match_type=True,
        similarity_threshold=0.88,
        max_workers=MAX_WORKERS,
        on_progress=on_progress,
    )

    elapsed = time.time() - t0
    created_n = sum(1 for r in node_results if r.created)
    merged_n = sum(1 for r in node_results if not r.created)
    created_e = sum(1 for r in edge_results if r.created)
    merged_e = sum(1 for r in edge_results if not r.created)

    print(f"\n  Done in {elapsed:.1f}s ({len(chunks) * 2} LLM calls)")
    print(f"  Nodes: {created_n} new, {merged_n} fuzzy-merged")
    print(f"  Edges: {created_e} new, {merged_e} deduped")

    # --- Post-extraction dedupe ---
    print(f"\n--- Post-extraction duplicate resolution ---")
    t1 = time.time()
    clusters = db.resolve_duplicates(
        match_keys=["name"],
        match_type=True,
        similarity_threshold=0.90,
    )
    print(f"  {len(clusters)} clusters resolved in {time.time() - t1:.1f}s")
    for c in clusters[:10]:
        merged_names = [m.properties.get("name", "?") for m in c.merged]
        print(f"    '{c.survivor.properties.get('name')}' <- {merged_names}")
    if len(clusters) > 10:
        print(f"    ... and {len(clusters) - 10} more")

    # --- Results ---
    all_nodes = db.nodes_to_json()
    all_edges = db.edges_to_json()

    type_counts = {}
    for n in all_nodes:
        t = n.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n--- Final Graph ---")
    print(f"  Nodes: {len(all_nodes)}")
    print(f"  Edges: {len(all_edges)}")
    print(f"  Types: {dict(sorted(type_counts.items(), key=lambda x: -x[1]))}")

    print(f"\n--- Sample Entities (first 30) ---")
    sorted_nodes = sorted(all_nodes, key=lambda x: (x.get("type") or "", x.get("properties", {}).get("name") or ""))
    for n in sorted_nodes[:30]:
        props = n.get("properties", {})
        print(f"  [{n.get('type', '?'):15}] {props.get('name', props)}")
    if len(sorted_nodes) > 30:
        print(f"  ... and {len(sorted_nodes) - 30} more")

    print(f"\n--- Sample Relationships (first 20) ---")
    node_id_map = {n["id"]: n for n in all_nodes}
    for e in all_edges[:20]:
        src = node_id_map.get(e["source_id"], {}).get("properties", {}).get("name", "?")
        tgt = node_id_map.get(e["target_id"], {}).get("properties", {}).get("name", "?")
        print(f"  {src} --[{e['relation']}]--> {tgt}")
    if len(all_edges) > 20:
        print(f"  ... and {len(all_edges) - 20} more")

    print(f"\n--- Search: 'artificial intelligence' ---")
    results = db.search_nodes("artificial intelligence", limit=5)
    for sr in results:
        print(f"  [{sr.node.type}] {sr.node.properties.get('name', '?')} (score={sr.score:.3f})")

    print(f"\n{'=' * 60}")
    print(f"{len(all_nodes)} nodes, {len(all_edges)} edges from {len(text):,} chars in {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
