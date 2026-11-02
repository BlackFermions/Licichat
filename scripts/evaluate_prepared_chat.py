"""Bounded opt-in eval: existing evidence only, read-only SQL, no ingestion/quotas.

Fact-anchor coverage and citation allow-list checks are regression proxies, NOT
an automatic claim of answer correctness or exhaustive requirement coverage.
"""

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
import unicodedata
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def normalize(text):
    return " ".join("".join(c for c in unicodedata.normalize("NFKD", str(text).lower())
                            if not unicodedata.combining(c)).split())


def cases():
    dataset = json.loads((ROOT / "evals/prepared_chat_cases.json").read_text(encoding="utf-8"))
    for family in dataset["families"]:
        for index, question in enumerate(family["questions"]):
            yield {**family, "id": f"{family['id']}-{index + 1}", "family": family["id"],
                   "question": question, "sample_answer": index == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Autoriza llamadas acotadas de embeddings/chat")
    parser.add_argument("--azure-readonly", action="store_true", help="Lee configuracion actual de Azure en memoria")
    parser.add_argument("--limit", type=int, default=50, choices=range(1, 51))
    parser.add_argument("--answers", type=int, default=14, choices=range(0, 15))
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--only-family", help="Subconjunto explicito para regresiones dirigidas")
    parser.add_argument("--answer-each", action="store_true", help="Responde cada variante seleccionada, manteniendo --answers como limite")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.live:
        parser.error("Requiere --live; consume API. No ejecuta ingesta, correos ni cuotas de usuarios.")
    os.environ["PGOPTIONS"] = "-c default_transaction_read_only=on -c statement_timeout=15000"
    image = None
    if args.azure_readonly:
        from scripts.evaluation_runtime import load_azure_environment
        image = load_azure_environment()
    import lite_chat_engine as chat
    import pilot_chat_retrieval as retrieval

    results = []
    answer_count = 0
    selected_cases = [case for case in cases() if not args.only_family or case['family'] == args.only_family]
    if not selected_cases:
        parser.error('Familia sin casos')
    for case in selected_cases[:args.limit]:
        started = time.monotonic()
        try:
            found = retrieval.retrieve_pilot(case["tender_id"], case["question"], chat.openai_client)
        except Exception as exc:
            found = None
            error_type = type(exc).__name__
        else:
            error_type = None
        elapsed = round((time.monotonic() - started) * 1000)
        status, context, references = found if found else ({}, "", [])
        checks = [bool(re.search(pattern, normalize(context))) for pattern in case["patterns"]]
        selected = status.get("selected_document_roles") or [status.get("selected_document_role")]
        roles_ok = set(case["roles"]).issubset(set(selected))
        item = {"id": case["id"], "family": case["family"], "tender_id": case["tender_id"],
                "question": case["question"], "retrieval_ms": elapsed,
                "patterns_found": checks, "roles_ok": roles_ok,
                "evidence_pass": bool(found) and all(checks) and roles_ok if checks else None,
                "context_chars": len(context), "references": references, "error_type": error_type}
        if found and (case["sample_answer"] or args.answer_each) and answer_count < args.answers:
            # Same app/prompt/model as production, but reuse measured retrieval and
            # explicitly prohibit the downloader: failures cannot trigger ingestion.
            started = time.monotonic()
            try:
                with patch.object(chat, "retrieve_pilot", return_value=found), \
                     patch.object(chat, "prepare_corpus", side_effect=RuntimeError("eval_download_forbidden")), \
                     patch.object(chat, "SERVICE_KEY", "local-evaluation-not-a-secret"):
                    response = chat.app.test_client().post("/api/v1/chat_stream",
                        headers={"X-Service-Key": "local-evaluation-not-a-secret"},
                        json={"tender_id": case["tender_id"], "message": case["question"],
                              "use_document_pilot": True})
                    answer = response.get_data(as_text=True)
                item["answer"] = answer
                item["answer_ms"] = round((time.monotonic() - started) * 1000)
                patterns = case.get("answer_patterns", case["patterns"])
                item["answer_anchor_pass"] = all(re.search(p, normalize(answer)) for p in patterns)
                # Unsupported fabricated/absent citations are counted separately.
                citations = re.findall(r"\[([^\[\]]+),\s*p(?:p)?\.?\s*(\d+(?:-\d+)?)\]", answer)
                allowed = {(normalize(r["document"]), str(r["page"])) for r in references}
                item["citation_count"] = len(citations)
                item["citations_allowed"] = all((normalize(d), p) in allowed for d, p in citations)
            except Exception as exc:
                item["answer_error"] = type(exc).__name__
            answer_count += 1
        results.append(item)
        print(json.dumps({k: item[k] for k in ("id", "evidence_pass", "retrieval_ms", "error_type")}), flush=True)

    scored = [r for r in results if r["evidence_pass"] is not None]
    answered = [r for r in results if "answer_anchor_pass" in r]
    times = sorted(r["retrieval_ms"] for r in results)
    summary = {"cases": len(results), "evidence_cases": len(scored),
               "evidence_pass": sum(r["evidence_pass"] for r in scored),
               "answers": len(answered), "answer_anchor_pass": sum(r["answer_anchor_pass"] for r in answered),
               "retrieval_median_ms": statistics.median(times),
               "retrieval_p95_ms": times[min(len(times)-1, int(.95*len(times)))],
               "scope": "Local production code/model, live read-only DB; no downloads. Answer time excludes retrieval.",
               "limitations": "Anchor checks are not semantic correctness. Paraphrases share fact families; PDF audit is separate."}
    report = {"label": args.label, "runtime_image": image, "summary": summary, "results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
