import json

d = json.load(open("evaluation/retrieval_50_results.json", encoding="utf-8"))
test = {t["query_id"]: t for t in json.load(open("evaluation/test_set_50.json", encoding="utf-8"))}
sp = d["results"]["Sparse (BM25Okapi)"]["per_query"]
miss = [q for q in sp if q["hit@5"] == 0]
print(f"Missed at @5: {len(miss)} / 50")
for q in miss:
    t = test[q["query_id"]]
    print(f"- {q['query_id']} ({t['expected_section_id']}): {t['query'][:90]}...")
    print(f"  expected: {t['expected_chunk_ids']}")
    print(f"  got top5: {q['top_retrieved']}")