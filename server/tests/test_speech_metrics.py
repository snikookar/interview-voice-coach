from analysis.speech_metrics import (
    TurnWindow,
    Word,
    compute_metrics,
    count_fillers,
    talk_ratio,
    words_from_text,
)


def _words(text: str, start: float = 0.0, step: float = 0.4) -> list[Word]:
    return [Word(t, start + i * step, start + i * step + 0.3) for i, t in enumerate(text.split())]


def test_counts_hesitations_and_phrases():
    counts = count_fillers(
        _words("Um, so I, uh, basically used it, you know, for search. I mean it worked.")
    )
    assert counts == {"um": 1, "uh": 1, "basically": 1, "you know": 1, "i mean": 1}


def test_like_is_only_a_filler_when_set_off():
    # Comparison: "metrics like MRR" is not a filler.
    assert count_fillers(_words("I'd use ranking metrics like MRR.")) == {}
    # Hedge: "it was, like, slow" is.
    assert count_fillers(_words("It was, like, really slow.")) == {"like": 1}


def test_kind_of_only_counts_as_a_hedge():
    assert count_fillers(_words("It is a kind of index.")) == {}
    assert count_fillers(_words("It was kind of, slow.")) == {"kind of": 1}


def test_wpm_pauses_and_response_delay():
    # 30 words at 0.4 s spacing (about 12 s), then a 4 s thinking pause, then 10 more words.
    first = _words(" ".join(["word"] * 30), start=10.0)
    second = _words(" ".join(["word"] * 10), start=first[-1].end + 4.0)
    window = TurnWindow(
        start_ms=9_500, end_ms=int(second[-1].end * 1000), question_id="q1", prev_bot_end_ms=8_000
    )
    m = compute_metrics(first + second, [window])

    assert m.words == 40
    assert m.long_pauses == 1 and m.longest_pause_secs == 4.0
    # The 4 s pause is excluded from speaking time: 40 words over about 15.9 s.
    assert 140 <= m.wpm <= 160
    assert m.response_delays_secs == [2.0]  # bot finished at 8.0 s, first word at 10.0 s


def test_words_outside_turn_windows_are_ignored():
    words = _words("hello there", start=100.0)
    m = compute_metrics(words, [TurnWindow(0, 5_000, "q1")])
    assert m.words == 0 and m.wpm is None


def test_talk_ratio_and_text_fallback():
    windows = [TurnWindow(0, 30_000, "q1"), TurnWindow(40_000, 50_000, "q2")]
    assert talk_ratio(windows, [(30_000, 40_000)]) == 4.0
    words = words_from_text("one two three four", 1_000, 3_000)
    assert [round(w.start, 2) for w in words] == [1.0, 1.5, 2.0, 2.5]
