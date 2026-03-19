"""End-to-end test: ingest aimav4.txt using real LLM extraction via DSPy."""

import sys
import os
import re
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dspy
from graphmemory import GraphMemory, MergeStrategy
from graphmemory.extraction import extract_and_merge

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Configure DSPy with gpt-5-nano ---
lm = dspy.LM("openai/gpt-5-nano")
dspy.configure(lm=lm)


def chunk_text(text: str, max_chars: int = 3000) -> list[str]:
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


def main():
    input_path = os.path.join(os.path.dirname(__file__), "..", "input", "aimav4.txt")
    with open(input_path) as f:
        text = f.read(100_000)

    text = re.sub(r"<!--.*?-->", "", text)
    chunks = chunk_text(text, max_chars=4000)

    print("=" * 60)
    print("GraphMemory — Real LLM Extraction Test")
    print("=" * 60)
    print(f"Source: aimav4.txt ({len(text)} chars)")
    print(f"Chunks: {len(chunks)}")
    print(f"LLM: gpt-5-nano via DSPy")

    db = GraphMemory(database=":memory:", vector_length=3)

    print(f"\n--- Extracting entities & relationships ---")
    total_nodes = 0
    total_edges = 0
    total_merged_nodes = 0
    total_merged_edges = 0

    for i, chunk in enumerate(chunks):
        print(f"\n  Chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)...")
        try:
            # Pass each chunk as a single "sentence" to avoid per-sentence LLM calls
            node_results, edge_results = extract_and_merge(
                db,
                chunk,
                match_keys=["name"],
                match_type=True,
                similarity_threshold=0.88,
                sentences=[chunk],  # single LLM call per chunk
            )
            created_n = sum(1 for r in node_results if r.created)
            merged_n = sum(1 for r in node_results if not r.created)
            created_e = sum(1 for r in edge_results if r.created)
            merged_e = sum(1 for r in edge_results if not r.created)

            total_nodes += created_n
            total_merged_nodes += merged_n
            total_edges += created_e
            total_merged_edges += merged_e

            print(f"    Nodes: {created_n} new, {merged_n} merged")
            print(f"    Edges: {created_e} new, {merged_e} merged")
        except Exception as e:
            logger.warning(f"  Chunk {i + 1} failed: {e}")

    # --- Post-extraction dedupe ---
    print(f"\n--- Post-extraction duplicate resolution ---")
    clusters = db.resolve_duplicates(
        match_keys=["name"],
        match_type=True,
        similarity_threshold=0.90,
    )
    if clusters:
        for c in clusters:
            merged_names = [m.properties.get("name", "?") for m in c.merged]
            print(f"  Merged: '{c.survivor.properties.get('name')}' <- {merged_names}")
    else:
        print("  No additional duplicates found.")

    # --- Results ---
    all_nodes = db.nodes_to_json()
    all_edges = db.edges_to_json()

    print(f"\n--- Final Graph ---")
    print(f"  Nodes: {len(all_nodes)}")
    print(f"  Edges: {len(all_edges)}")

    type_counts = {}
    for n in all_nodes:
        t = n.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  Types: {type_counts}")

    print(f"\n--- Extracted Entities ---")
    for n in sorted(all_nodes, key=lambda x: (x.get("type", ""), x.get("properties", {}).get("name", ""))):
        props = n.get("properties", {})
        print(f"  [{n.get('type', '?'):15}] {props.get('name', props)}")

    print(f"\n--- Extracted Relationships ---")
    node_id_map = {n["id"]: n for n in all_nodes}
    for e in all_edges:
        src = node_id_map.get(e["source_id"], {}).get("properties", {}).get("name", e["source_id"])
        tgt = node_id_map.get(e["target_id"], {}).get("properties", {}).get("name", e["target_id"])
        print(f"  {src} --[{e['relation']}]--> {tgt}")

    print(f"\n--- Full-text search: 'deep learning' ---")
    results = db.search_nodes("deep learning", limit=5)
    for sr in results:
        print(f"  [{sr.node.type}] {sr.node.properties.get('name', '?')} (score={sr.score:.3f})")

    print(f"\n--- Summary ---")
    print(f"  Extracted: {total_nodes} nodes, {total_edges} edges")
    print(f"  Fuzzy-merged during ingest: {total_merged_nodes} nodes, {total_merged_edges} edges")
    print(f"  Post-dedupe clusters: {len(clusters)}")
    print(f"  Final graph: {len(all_nodes)} nodes, {len(all_edges)} edges")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
