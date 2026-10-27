"""End-to-end check of reword -> human review -> finalize, with the API call faked."""
import csv
import json

import src.make_variants as mv


def test_reword_review_finalize(tmp_path, monkeypatch):
    monkeypatch.setattr(mv, "DATA", tmp_path)
    (tmp_path / "reworded").mkdir()
    items = tmp_path / "toy.jsonl"
    rows = [
        {"item_id": "a", "question": "What is 2+3 in base 10?", "options": ["5", "6"]},
        {"item_id": "b", "question": "Name the capital of France.", "options": ["Paris", "Rome"]},
    ]
    items.write_text("\n".join(json.dumps(r) for r in rows))
    fake = {"What is 2+3 in base 10?": "In base 10, what does 2+3 equal?",
            "Name the capital of France.": "Which city is the capital of France?"}
    monkeypatch.setattr(mv, "call_chat", lambda q, k: fake[q])

    mv.cmd_reword(items, n_rewordings=2)
    sheet = tmp_path / "reworded" / "toy_review.csv"
    with open(sheet) as f:
        review = list(csv.DictReader(f))
    assert len(review) == 4 and all(r["auto_issues"] == "" for r in review)

    for r in review:  # reviewer accepts item a, rejects one rewording of b
        r["human_ok"] = "0" if (r["item_id"] == "b" and r["wording_id"] == "w2") else "1"
    with open(sheet, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=review[0].keys()); w.writeheader(); w.writerows(review)

    mv.cmd_finalize(items, n_rewordings=2)
    out = [json.loads(l) for l in open(tmp_path / "reworded" / "toy_wordings.jsonl")]
    assert {r["item_id"] for r in out} == {"a"}          # b dropped to keep the design fully crossed
    assert [r["wording_id"] for r in out] == ["w0", "w1", "w2"]
