import argparse
import json
from pathlib import Path
import sys

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.rag import search_records


QUESTIONS = [
    "What programs are offered by the Computing department?",
    "List faculty members in the Electrical Engineering department.",
    "Who is the Head of Department of Computing?",
    "Give me all MS programs mentioned in the knowledge base.",
    "Does IST offer PhD in Electrical Engineering?",
    "What are BS programs in Space Science department?",
    "Who are the professors in Mechanical Engineering?",
    "What is the contact email of the Vice Chancellor?",
    "What is the fee refund policy timeline?",
    "What scholarships are available for foreign students?",
    "Does IST conduct its own entry test?",
    "What is the minimum entry test score for BS applications?",
    "List all departments mentioned in faculty and programs section.",
    "Who is the HOD of Electrical Engineering?",
    "Name assistant professors in Computing department.",
    "Which graduate programs are available in Space Science?",
    "Does Computing offer PhD program in the listed department block?",
    "What is the admission schedule for graduate spring 2026?",
    "What is the duration of BS programs?",
    "Can pre-medical students apply for engineering programs?",
    "List programs in Applied Mathematics and Statistics.",
    "Who is Director QEC?",
    "What are labs and facilities in Computing?",
    "Which departments have postgraduate PhD listed?",
    "Give complete list of degree programs including BS MS and PhD.",
]


def build_answer(client: OpenAI, question: str, context: str, model: str) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=220,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the provided context. "
                    "If evidence is missing, say: I do not have that information in the IST knowledge base."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}",
            },
        ],
    )
    return response.choices[0].message.content or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IST RAG retrieval and grounded answers.")
    parser.add_argument("--k", type=int, default=5, help="Top-k contexts to retrieve.")
    parser.add_argument("--limit", type=int, default=len(QUESTIONS), help="How many questions to run.")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model for answer generation.")
    parser.add_argument("--output", default="data/eval_results.json", help="Output JSON file path.")
    parser.add_argument("--no-llm", action="store_true", help="Only run retrieval and skip answer generation.")
    args = parser.parse_args()

    root = ROOT
    load_dotenv(root / ".env")

    client = None if args.no_llm else OpenAI()
    selected_questions = QUESTIONS[: max(1, min(args.limit, len(QUESTIONS)))]

    results = []
    for i, question in enumerate(selected_questions, start=1):
        contexts = search_records(question, k=args.k)
        context_text = "\n\n".join(c["text"] for c in contexts)
        answer = ""
        if client is not None:
            answer = build_answer(client, question, context_text, args.model)

        print(f"\n[{i}] Question: {question}")
        print("Retrieved context (preview):")
        print(context_text[:900])
        if answer:
            print("Answer:")
            print(answer)

        results.append(
            {
                "question": question,
                "retrieved_contexts": contexts,
                "concatenated_context": context_text,
                "answer": answer,
            }
        )

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation output to: {output_path}")


if __name__ == "__main__":
    main()
