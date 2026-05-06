import argparse
import difflib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TEXTS_PATH = DATA_DIR / "texts.json"
KB_PATH = DATA_DIR / "KB.txt"
CHUNKS_PATH = DATA_DIR / "chunks.json"
INDEX_PATH = DATA_DIR / "index.faiss"
EMBED_MODEL = "text-embedding-3-small"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_RERANKER_DEFAULT = os.getenv("RAG_USE_RERANKER", "true").lower() in {"1", "true", "yes", "on"}
MIN_CHUNK_CHARS = 80
EMBED_BATCH_SIZE = 100
client = None
cross_encoder = None
MATCH_STOPWORDS = {
	"a", "an", "the", "is", "are", "was", "were", "who", "what", "when", "where", "why", "how",
	"about", "of", "in", "on", "for", "to", "from", "and", "or", "with", "at", "by", "my", "me",
	"tell", "please", "kindly", "dr", "mr", "ms", "mrs", "prof", "professor"
}


@dataclass
class ChunkRecord:
	id: str
	text: str
	metadata: dict[str, Any]


def get_client():
	global client
	if client is None:
		client = OpenAI()
	return client


def get_cross_encoder():
	global cross_encoder
	if cross_encoder is None:
		cross_encoder = CrossEncoder(RERANK_MODEL)
	return cross_encoder


def preload_models(preload_reranker: bool = True) -> None:
	if preload_reranker and USE_RERANKER_DEFAULT:
		get_cross_encoder()


def normalize_text(text: str) -> str:
	return re.sub(r"\s+", " ", text).strip()


def read_text_file(path: Path) -> str:
	raw = path.read_bytes()
	for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
		try:
			text = raw.decode(encoding)
			if "�" not in text:
				return text
		except UnicodeDecodeError:
			continue
	return raw.decode("utf-8", errors="replace")


def split_long_text(text: str, size: int = 1200, overlap: int = 150):
	chunks = []
	text = normalize_text(text)
	if not text:
		return chunks
	if size <= overlap:
		raise ValueError("size must be greater than overlap")

	step = size - overlap
	for i in range(0, len(text), step):
		chunks.append(text[i:i + size])
	return chunks


def parse_sections(text: str):
	sections = []
	current_title = "General"
	buffer = []

	for line in text.splitlines():
		if line.startswith("## "):
			if buffer:
				sections.append((current_title, "\n".join(buffer).strip()))
			buffer = []
			current_title = line[3:].strip()
		else:
			buffer.append(line)

	if buffer:
		sections.append((current_title, "\n".join(buffer).strip()))

	return sections


def infer_content_type(text: str) -> str:
	t = text.lower()
	if "department:" in t and "programs offered by this department" in t:
		return "department_profile"
	if t.startswith("q:") or ("q:" in t and "a:" in t):
		return "faq"
	if "table 1:" in t:
		return "table"
	if t.startswith("---") and t.endswith("---"):
		return "heading"
	return "narrative"


def extract_blocks(section_body: str) -> list[tuple[str, str]]:
	lines = [ln.rstrip() for ln in section_body.splitlines()]
	blocks: list[tuple[str, str]] = []

	current_heading = ""
	buffer: list[str] = []

	def flush_buffer():
		nonlocal buffer
		body = "\n".join(buffer).strip()
		buffer = []
		if body:
			blocks.append((current_heading, body))

	i = 0
	while i < len(lines):
		line = lines[i].strip()
		if not line or set(line) == {"="}:
			flush_buffer()
			i += 1
			continue

		if line.startswith("---") and line.endswith("---"):
			flush_buffer()
			current_heading = line.strip("-").strip()
			i += 1
			continue

		if line.startswith("DEPARTMENT:"):
			flush_buffer()
			dept_lines = [line]
			i += 1
			while i < len(lines):
				nxt = lines[i].strip()
				if not nxt:
					dept_lines.append("")
					i += 1
					continue
				if nxt.startswith("DEPARTMENT:"):
					break
				if nxt.startswith("---") and nxt.endswith("---"):
					break
				if set(nxt) == {"="}:
					i += 1
					continue
				dept_lines.append(nxt)
				i += 1
			body = "\n".join(dept_lines).strip()
			if body:
				blocks.append((current_heading, body))
			continue

		buffer.append(line)
		i += 1

	flush_buffer()
	return blocks


def format_record_text(section_title: str, heading: str, body: str) -> str:
	parts = [f"SECTION: {section_title}"]
	if heading:
		parts.append(f"HEADING: {heading}")
	parts.append(body)
	return "\n".join(parts)


def build_kb_chunk_records(text: str) -> list[ChunkRecord]:
	records: list[ChunkRecord] = []
	for sec_idx, (section_title, section_body) in enumerate(parse_sections(text)):
		blocks = extract_blocks(section_body)
		for block_idx, (heading, body) in enumerate(blocks):
			norm_body = body.strip()
			if not norm_body:
				continue

			text_chunk = format_record_text(section_title, heading, norm_body)
			content_type = infer_content_type(norm_body)
			metadata = {
				"section": section_title,
				"heading": heading,
				"content_type": content_type,
			}

			if len(text_chunk) <= 1600:
				records.append(
					ChunkRecord(
						id=f"s{sec_idx:03d}_b{block_idx:04d}",
						text=text_chunk,
						metadata=metadata,
					)
				)
			else:
				for piece_idx, piece in enumerate(split_long_text(text_chunk, size=1300, overlap=180)):
					records.append(
						ChunkRecord(
							id=f"s{sec_idx:03d}_b{block_idx:04d}_p{piece_idx:02d}",
							text=piece,
							metadata=metadata,
						)
					)

	seen = set()
	unique_records: list[ChunkRecord] = []
	for rec in records:
		if len(rec.text) < MIN_CHUNK_CHARS:
			continue
		if rec.text in seen:
			continue
		seen.add(rec.text)
		unique_records.append(rec)

	return unique_records


def build_json_chunks(data):
	chunks = []
	for item in data:
		url = item.get("url", "Unknown source")
		text = item.get("text", "")
		for piece in split_long_text(text, size=1200, overlap=150):
			chunks.append(f"Source: {url}\n{piece}")
	return chunks


def build_json_chunk_records(data) -> list[ChunkRecord]:
	records: list[ChunkRecord] = []
	for item_idx, item in enumerate(data):
		url = item.get("url", "Unknown source")
		text = item.get("text", "")
		for piece_idx, piece in enumerate(split_long_text(text, size=1200, overlap=150)):
			records.append(
				ChunkRecord(
					id=f"j{item_idx:04d}_p{piece_idx:02d}",
					text=f"Source: {url}\n{piece}",
					metadata={"section": "json_source", "heading": url, "content_type": "narrative"},
				)
			)
	return records


def load_source_chunks(source: str = "auto"):
	if source in ("auto", "txt") and KB_PATH.exists() and read_text_file(KB_PATH).strip():
		text = read_text_file(KB_PATH)
		records = build_kb_chunk_records(text)
		return records, "txt"

	if source in ("auto", "json") and TEXTS_PATH.exists():
		with TEXTS_PATH.open("r", encoding="utf-8") as f:
			data = json.load(f)
		records = build_json_chunk_records(data)
		return records, "json"

	raise ValueError("No usable source found. Add data/KB.txt or populate data/texts.json.")


def embed(texts):
	all_embeddings = []
	for start in range(0, len(texts), EMBED_BATCH_SIZE):
		batch = texts[start:start + EMBED_BATCH_SIZE]
		res = get_client().embeddings.create(model=EMBED_MODEL, input=batch)
		all_embeddings.extend([r.embedding for r in res.data])
	return all_embeddings


def tokenize_for_bm25(text: str) -> list[str]:
	return re.findall(r"[a-z0-9]+", text.lower())


def tokenize_for_match(text: str):
	return [
		tok for tok in re.findall(r"[a-z0-9]+", text.lower())
		if len(tok) > 1 and tok not in MATCH_STOPWORDS
	]


def query_ngrams(tokens: list[str], n: int = 2) -> set[str]:
	if len(tokens) < n:
		return set()
	return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def extract_name_query_tokens(query: str):
	tokens = tokenize_for_match(query)
	# For person lookup, keep 2-4 most specific tokens (e.g., benish amin).
	if len(tokens) >= 2:
		return tokens[-4:]
	return tokens


def has_fuzzy_name_match(name_tokens, chunk_tokens) -> bool:
	if not name_tokens:
		return False

	for token in name_tokens:
		if token in chunk_tokens:
			continue
		best = max((difflib.SequenceMatcher(None, token, cand).ratio() for cand in chunk_tokens), default=0.0)
		if best < 0.78:
			return False
	return True


def heuristic_score(
	query_tokens: set[str],
	name_tokens: list[str],
	chunk: str,
	vector_rank: int,
) -> float:
	chunk_l = chunk.lower()
	chunk_tokens = set(tokenize_for_match(chunk))

	# Vector rank score is stable regardless of embedding distance scale.
	vector_rank_score = 1.0 / (vector_rank + 1)
	overlap = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
	name_bonus = 1.0 if name_tokens and all(tok in chunk_l for tok in name_tokens) else 0.0
	fuzzy_name_bonus = 0.8 if has_fuzzy_name_match(name_tokens, chunk_tokens) else 0.0
	return vector_rank_score + (1.5 * overlap) + (1.0 * name_bonus) + fuzzy_name_bonus


def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = 60) -> dict[int, float]:
	scores: dict[int, float] = {}
	for doc_list in ranked_lists:
		for rank, doc_id in enumerate(doc_list):
			scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
	return scores


def rerank_candidates(query: str, candidate_ids: list[int], texts: list[str]) -> list[int]:
	if not candidate_ids:
		return []

	model = get_cross_encoder()
	pairs = [(query, texts[doc_id]) for doc_id in candidate_ids]
	scores = model.predict(pairs)
	sorted_idx = np.argsort(-np.array(scores))
	return [candidate_ids[i] for i in sorted_idx]


def build_index(source: str = "auto"):
	records, selected_source = load_source_chunks(source)
	texts = [r.text for r in records]

	if not texts:
		raise ValueError("No text chunks found in the selected source.")

	embeddings = embed(texts)

	dim = len(embeddings[0])
	index = faiss.IndexFlatL2(dim)
	index.add(np.array(embeddings).astype("float32"))

	DATA_DIR.mkdir(parents=True, exist_ok=True)
	faiss.write_index(index, str(INDEX_PATH))
	with CHUNKS_PATH.open("w", encoding="utf-8") as f:
		json.dump(
			{
				"schema_version": 2,
				"source": selected_source,
				"chunks": [{"id": r.id, "text": r.text, "metadata": r.metadata} for r in records],
			},
			f,
			ensure_ascii=False,
		)

	print(f"Index built from {selected_source} with {len(texts)} chunks.")


def _load_records_and_texts():
	with CHUNKS_PATH.open("r", encoding="utf-8") as f:
		payload = json.load(f)

	if isinstance(payload, dict) and payload.get("schema_version") == 2:
		records = payload["chunks"]
		texts = [r["text"] for r in records]
		return records, texts

	legacy_chunks = payload["chunks"] if isinstance(payload, dict) else payload
	records = [
		{"id": f"legacy_{i}", "text": txt, "metadata": {"section": "legacy", "heading": "", "content_type": "narrative"}}
		for i, txt in enumerate(legacy_chunks)
	]
	texts = [r["text"] for r in records]
	return records, texts


def search_records(
	query: str,
	k: int = 5,
	dense_candidate_k: int = 120,
	bm25_candidate_k: int = 120,
	rerank_k: int = 40,
	use_reranker: bool | None = None,
):
	if use_reranker is None:
		use_reranker = USE_RERANKER_DEFAULT

	index = faiss.read_index(str(INDEX_PATH))
	records, texts = _load_records_and_texts()

	if not texts:
		return []

	q_emb = embed([query])[0]
	dense_k = min(max(dense_candidate_k, k * 20), len(texts))
	_, dense_indices = index.search(np.array([q_emb]).astype("float32"), k=dense_k)
	dense_ranked = [int(idx) for idx in dense_indices[0] if 0 <= idx < len(texts)]

	tokenized_docs = [tokenize_for_bm25(t) for t in texts]
	bm25 = BM25Okapi(tokenized_docs)
	q_tokens = tokenize_for_bm25(query)
	bm25_scores = bm25.get_scores(q_tokens)
	bm25_ranked = [int(i) for i in np.argsort(-bm25_scores)[: min(bm25_candidate_k, len(texts))]]

	rrf_scores = reciprocal_rank_fusion([dense_ranked, bm25_ranked], k=60)
	fused_ranked = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)

	if use_reranker and rerank_k > 0:
		initial = fused_ranked[: min(max(rerank_k, k * 8), len(fused_ranked))]
		reranked = rerank_candidates(query, initial, texts)
		selected_ids = reranked[:k]
	else:
		selected_ids = fused_ranked[:k]
	return [records[idx] for idx in selected_ids]


def search(query, k: int = 3):
	records = search_records(query, k=k)
	texts = [r["text"][:1600] for r in records]
	return "\n\n".join(texts)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Build or query the FAISS index.")
	parser.add_argument("command", choices=["build", "search"], nargs="?", default="build")
	parser.add_argument("--source", choices=["auto", "txt", "json"], default="auto")
	parser.add_argument("--query", help="Query text for search mode")
	parser.add_argument("--k", type=int, default=3, help="Top-k results for search mode")
	args = parser.parse_args()

	if args.command == "build":
		build_index(source=args.source)
	else:
		if not args.query:
			raise ValueError("--query is required for search mode")
		print(search(args.query, k=args.k))
