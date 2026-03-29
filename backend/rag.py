import argparse
import json
import re
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TEXTS_PATH = DATA_DIR / "texts.json"
KB_PATH = DATA_DIR / "KB.txt"
CHUNKS_PATH = DATA_DIR / "chunks.json"
INDEX_PATH = DATA_DIR / "index.faiss"
EMBED_MODEL = "text-embedding-3-small"
MIN_CHUNK_CHARS = 80
EMBED_BATCH_SIZE = 100
client = None
MATCH_STOPWORDS = {
	"a", "an", "the", "is", "are", "was", "were", "who", "what", "when", "where", "why", "how",
	"about", "of", "in", "on", "for", "to", "from", "and", "or", "with", "at", "by", "my", "me",
	"tell", "please", "kindly", "dr", "mr", "ms", "mrs", "prof", "professor"
}


def get_client():
	global client
	if client is None:
		client = OpenAI()
	return client


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


def build_kb_chunks(text: str):
	chunks = []
	for section_title, section_body in parse_sections(text):
		current_subsection = ""
		paragraph_buffer = []

		def flush_paragraphs():
			nonlocal paragraph_buffer
			paragraph_text = normalize_text(" ".join(paragraph_buffer))
			paragraph_buffer = []
			if not paragraph_text:
				return

			prefix = section_title
			if current_subsection:
				prefix = f"{prefix} | {current_subsection}"

			combined = f"{prefix}\n{paragraph_text}"
			if len(combined) <= 1400:
				chunks.append(combined)
			else:
				for piece in split_long_text(combined, size=1200, overlap=150):
					chunks.append(piece)

		for raw_line in section_body.splitlines():
			line = raw_line.strip()
			if not line or set(line) == {"="}:
				flush_paragraphs()
				continue

			if line.startswith("===") and line.endswith("==="):
				flush_paragraphs()
				current_subsection = line.strip("= ")
				continue

			# FAQ-style knowledge is best retrieved as self-contained question-answer chunks.
			if "?" in line and len(line) <= 1800:
				flush_paragraphs()
				prefix = section_title
				if current_subsection:
					prefix = f"{prefix} | {current_subsection}"
				chunks.append(f"{prefix}\n{normalize_text(line)}")
				continue

			paragraph_buffer.append(line)

		flush_paragraphs()

	# Remove duplicates while preserving order.
	seen = set()
	unique_chunks = []
	for chunk in chunks:
		if len(chunk) < MIN_CHUNK_CHARS:
			continue
		if chunk not in seen:
			seen.add(chunk)
			unique_chunks.append(chunk)

	return unique_chunks


def build_json_chunks(data):
	chunks = []
	for item in data:
		url = item.get("url", "Unknown source")
		text = item.get("text", "")
		for piece in split_long_text(text, size=1200, overlap=150):
			chunks.append(f"Source: {url}\n{piece}")
	return chunks


def load_source_chunks(source: str = "auto"):
	if source in ("auto", "txt") and KB_PATH.exists() and read_text_file(KB_PATH).strip():
		text = read_text_file(KB_PATH)
		return build_kb_chunks(text), "txt"

	if source in ("auto", "json") and TEXTS_PATH.exists():
		with TEXTS_PATH.open("r", encoding="utf-8") as f:
			data = json.load(f)
		return build_json_chunks(data), "json"

	raise ValueError("No usable source found. Add data/KB.txt or populate data/texts.json.")


def embed(texts):
	all_embeddings = []
	for start in range(0, len(texts), EMBED_BATCH_SIZE):
		batch = texts[start:start + EMBED_BATCH_SIZE]
		res = get_client().embeddings.create(model=EMBED_MODEL, input=batch)
		all_embeddings.extend([r.embedding for r in res.data])
	return all_embeddings


def tokenize_for_match(text: str):
	return [
		tok for tok in re.findall(r"[a-z0-9]+", text.lower())
		if len(tok) > 1 and tok not in MATCH_STOPWORDS
	]


def extract_name_query_tokens(query: str):
	tokens = tokenize_for_match(query)
	# For person lookup, keep 2-4 most specific tokens (e.g., benish amin).
	if len(tokens) >= 2:
		return tokens[-4:]
	return tokens


def build_index(source: str = "auto"):
	texts, selected_source = load_source_chunks(source)

	if not texts:
		raise ValueError("No text chunks found in the selected source.")

	embeddings = embed(texts)

	dim = len(embeddings[0])
	index = faiss.IndexFlatL2(dim)
	index.add(np.array(embeddings).astype("float32"))

	DATA_DIR.mkdir(parents=True, exist_ok=True)
	faiss.write_index(index, str(INDEX_PATH))
	with CHUNKS_PATH.open("w", encoding="utf-8") as f:
		json.dump({"source": selected_source, "chunks": texts}, f, ensure_ascii=False)

	print(f"Index built from {selected_source} with {len(texts)} chunks.")


def search(query, k: int = 3):
	index = faiss.read_index(str(INDEX_PATH))
	with CHUNKS_PATH.open("r", encoding="utf-8") as f:
		payload = json.load(f)

	texts = payload["chunks"] if isinstance(payload, dict) else payload
	q_emb = embed([query])[0]
	candidate_k = min(max(k * 8, 30), len(texts))
	distances, indices = index.search(np.array([q_emb]).astype("float32"), k=candidate_k)

	query_tokens = set(tokenize_for_match(query))
	name_tokens = extract_name_query_tokens(query)

	rescored = []
	for rank, idx in enumerate(indices[0]):
		if not (0 <= idx < len(texts)):
			continue
		chunk = texts[idx]
		chunk_l = chunk.lower()
		chunk_tokens = set(tokenize_for_match(chunk))

		# Vector rank score is stable regardless of embedding distance scale.
		vector_rank_score = 1.0 / (rank + 1)
		overlap = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
		name_bonus = 1.0 if name_tokens and all(tok in chunk_l for tok in name_tokens) else 0.0
		score = vector_rank_score + (1.5 * overlap) + (1.0 * name_bonus)
		rescored.append((score, chunk))

	rescored.sort(key=lambda x: x[0], reverse=True)
	selected = []
	seen = set()
	for _, chunk in rescored:
		trimmed = chunk[:950]
		if trimmed in seen:
			continue
		seen.add(trimmed)
		selected.append(trimmed)
		if len(selected) >= k:
			break

	return "\n\n".join(selected)


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
