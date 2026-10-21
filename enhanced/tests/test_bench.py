from enhanced.bench.locomo_worker import _text_of, gold_windows
from enhanced.bench.metrics import aggregate, paired_bootstrap, question_metrics

WINDOWS = [
    {"session": 1, "first_dia_id": "D1:1", "last_dia_id": "D1:6"},
    {"session": 1, "first_dia_id": "D1:7", "last_dia_id": "D1:9"},
    {"session": 2, "first_dia_id": "D2:1", "last_dia_id": "D2:6"},
]


def test_gold_windows_maps_dia_ids_to_windows():
    assert gold_windows(["D1:3", "D1:7", "D2:6"], WINDOWS) == [0, 1, 2]
    assert gold_windows(["D1:9", "D1:9"], WINDOWS) == [1]


def test_gold_windows_ignores_malformed_and_missing():
    assert gold_windows(["garbage", "D3:1", ""], WINDOWS) == []


def test_question_metrics():
    m = question_metrics(gold=[2, 5], ranked=[1, 2, 3, 4, 5])
    assert m["recall@1"] == 0.0
    assert m["recall@3"] == 0.5
    assert m["all@3"] == 0.0
    assert m["recall@5"] == 1.0 and m["all@5"] == 1.0
    assert m["mrr"] == 0.5
    assert question_metrics([9], [1, 2])["mrr"] == 0.0


def test_aggregate_skips_questions_without_gold():
    records = [
        {"category": "a", "gold": [1], "ranked": [1], "latency_s": 0.1},
        {"category": "b", "gold": [], "ranked": [1], "latency_s": 0.2},
    ]
    out = aggregate(records)
    assert out["all"]["n"] == 1 and "b" not in out
    assert out["all"]["recall@1"] == 1.0


def test_paired_bootstrap_detects_consistent_gain():
    result = paired_bootstrap([0.0] * 50, [1.0] * 40 + [0.0] * 10)
    assert result["delta"] == 0.8 and result["ci95"][0] > 0
    assert result["wins"] == 40 and result["losses"] == 0


class _Scored:
    payload = {"text": "hello"}


def test_text_of_handles_retriever_shapes():
    assert _text_of(_Scored()) == "hello"
    assert _text_of({"text": "a"}) == "a"
    assert _text_of({"payload": {"text": "b"}}) == "b"
    assert _text_of((_Scored(), 0.3)) == "hello"


def test_within_budget_stops_at_the_first_overflowing_window():
    from enhanced.bench.budget import within_budget

    tokens = {1: 400, 2: 700, 3: 100}
    assert within_budget([1, 2, 3], tokens, 1000) == ([1], 400)
    assert within_budget([1, 3, 2], tokens, 1200) == ([1, 3, 2], 1200)
    assert within_budget([], tokens, 1000) == ([], 0)


def test_store_lock_serialises_holders(tmp_path):
    import threading
    import time

    from enhanced.bench.profiles import store_lock

    events = []

    def hold(name, delay):
        with store_lock(tmp_path / "store"):
            events.append(f"{name}+")
            time.sleep(delay)
            events.append(f"{name}-")

    first = threading.Thread(target=hold, args=("a", 0.3))
    first.start()
    time.sleep(0.05)
    second = threading.Thread(target=hold, args=("b", 0))
    second.start()
    first.join()
    second.join()
    assert events == ["a+", "a-", "b+", "b-"]


def test_longmemeval_rounds_pair_user_turns_with_replies_and_dates():
    from enhanced.bench.longmemeval_worker import category, rounds

    question = {
        "question_id": "q1_abs",
        "question_type": "multi-session",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 10:00"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "bye"},
            ],
            [{"role": "assistant", "content": "orphan"}, {"role": "user", "content": "again"}],
        ],
    }

    out = rounds(question)

    assert [r["session_id"] for r in out] == ["s1", "s1", "s2"]
    assert (
        out[0]["text"]
        == "Chat session s1, 2023/05/20 (Sat) 02:21, round 1.\nUser: hi\nAssistant: hello"
    )
    assert out[1]["text"].endswith("User: bye\nAssistant: ")
    assert category(question) == "abstention"


def test_longmemeval_sample_is_stratified_and_stable(monkeypatch):
    from enhanced.bench import lme_run

    questions = {
        f"{kind}{i}": {"question_id": f"{kind}{i}", "question_type": kind}
        for kind in ("a", "b")
        for i in range(5)
    }
    monkeypatch.setattr(lme_run, "load_questions", lambda: questions)

    first = lme_run.sample(2, seed=0)

    assert first == lme_run.sample(2, seed=0)
    assert sum(q.startswith("a") for q in first) == 2 and len(first) == 4
