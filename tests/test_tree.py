"""Önek ağacı avantajının vaatlerini kanıtlayan testler (saf Python).

    python -m pytest -q tests
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from calculator_demo import direct_answer, tool_answer  # noqa: E402

from prefix_tree_rl import build_tree, render_tree, token_advantages  # noqa: E402

SCENARIO_1 = [
    (tool_answer("0294"), 0.98),
    (tool_answer("0294"), 0.98),
    (tool_answer("0294"), 0.98),
    (direct_answer("0294"), 1.0),
    (direct_answer("0300"), 0.0),
    (direct_answer("0284"), 0.0),
    (direct_answer("?"), 0.5),
    (direct_answer("1294"), 0.0),
]


def _split(pairs):
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _adv_of(seq, adv, token, occurrence=0):
    hits = [a for t, a in zip(seq, adv) if t == token]
    return hits[occurrence]


def _mean(xs):
    return sum(xs) / len(xs)


def test_telescope_sum_equals_dr_grpo():
    """Her cevabın token avantajlarının toplamı R_i − grup ortalaması."""
    seqs, rewards = _split(SCENARIO_1)
    adv = token_advantages(seqs, rewards)
    m = _mean(rewards)
    for a, r in zip(adv, rewards):
        assert sum(a) == pytest.approx(r - m, abs=1e-9)


def test_telescope_holds_on_random_trees():
    """Ortak önekli rastgele EOS'lu dizilerde teleskop tutar (her cevabın kendi yaprağı var)."""
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(2, 12)
        body = {tuple(rng.choice("ab") for _ in range(rng.randint(0, 6))) for _ in range(n)}
        seqs = [(*s, "<eos>") for s in body]
        rewards = [rng.random() for _ in seqs]
        adv = token_advantages(seqs, rewards)
        m = _mean(rewards)
        for a, r in zip(adv, rewards):
            assert sum(a) == pytest.approx(r - m, abs=1e-9)


def test_prefix_of_another_answer_breaks_exact_telescope():
    """EOS yoksa "a", "ab"nin önekidir: kısa cevabın toplamı V('a' düğümü) − ortalama olur."""
    adv = token_advantages([["a"], ["a", "b"]], [1.0, 0.0])
    assert sum(adv[0]) == pytest.approx(0.5 - 0.5)  # R − ortalama olsaydı +0.5 olurdu
    assert sum(adv[1]) == pytest.approx(0.0 - 0.5)  # uzun cevap için yine tam
    fixed = token_advantages([["a", "<eos>"], ["a", "b", "<eos>"]], [1.0, 0.0])
    assert sum(fixed[0]) == pytest.approx(0.5)


def test_credit_goes_to_the_decision_and_shared_tokens_get_zero():
    """Senaryo 1: <call> +0.425, ezbere başlamak (A) −0.255, tool yolunun geri kalanı 0."""
    seqs, rewards = _split(SCENARIO_1)
    adv = token_advantages(seqs, rewards)
    assert adv[0][0] == pytest.approx(0.98 - 0.555)
    assert _adv_of(seqs[4], adv[4], "A") == pytest.approx(0.30 - 0.555)
    assert sum(abs(x) > 1e-12 for x in adv[0]) == 1


def test_lucky_guess_decision_is_penalized():
    """Ezberle doğru bulan cevap: 'A' kararı negatif, doğru rakam '9' +0.5."""
    seqs, rewards = _split(SCENARIO_1)
    adv = token_advantages(seqs, rewards)
    assert _adv_of(seqs[3], adv[3], "A") < 0
    assert _adv_of(seqs[3], adv[3], "9") == pytest.approx(0.5)


def test_copy_error_blame_lands_only_on_the_wrong_digit():
    """7 doğru + A0295: sadece '5' cezalanır, <call> dahil geri kalan her şey 0."""
    seqs = [tool_answer("0294")] * 7 + [tool_answer("0295")]
    rewards = [0.98] * 7 + [-0.02]
    adv = token_advantages(seqs, rewards)
    m = _mean(rewards)
    assert _adv_of(seqs[7], adv[7], "5") == pytest.approx(-0.02 - m)
    assert sum(abs(x) > 1e-12 for x in adv[7]) == 1
    assert adv[7][0] == 0.0  # <call>


def test_identical_answers_share_credit():
    """Aynı token dizisi farklı ödül aldıysa (stokastik ödül) yaprak ortalamayı tutar."""
    adv = token_advantages([["x"], ["x"], ["y"]], [1.0, 0.0, 0.5])
    assert adv[0] == adv[1]
    assert adv[0][0] == pytest.approx(0.5 - 0.5)  # V(x) = 0.5, kök = 0.5


def test_groups_are_independent():
    seqs, rewards = _split(SCENARIO_1)
    both = token_advantages(seqs + seqs, rewards + [1 - r for r in rewards], [0] * 8 + [1] * 8)
    alone = token_advantages(seqs, rewards)
    for a, b in zip(both[:8], alone):
        assert a == pytest.approx(b)


def test_group_std_scale():
    seqs, rewards = _split(SCENARIO_1)
    raw = token_advantages(seqs, rewards)
    scaled = token_advantages(seqs, rewards, scale="group_std")
    m = _mean(rewards)
    s = (sum((r - m) ** 2 for r in rewards) / (len(rewards) - 1)) ** 0.5
    for a, b in zip(raw, scaled):
        assert [x / (s + 1e-6) for x in a] == pytest.approx(b)


def test_loo_baseline_excludes_self():
    seqs, rewards = _split(SCENARIO_1)
    adv = token_advantages(seqs, rewards, loo=True)
    others = (sum(rewards) - rewards[0]) / 7
    assert adv[0][0] == pytest.approx(0.98 - others)


def test_loo_gives_zero_where_nobody_else_went():
    """loo: tek başına kalınan önekte baseline yok -> avantaj 0."""
    adv = token_advantages([["a", "b", "d"], ["a", "c"]], [1.0, 0.0], loo=True)
    # 'a': önce kök (kendisi hariç = 0.0), sonra 'a' düğümü (ikisinin ortalaması 0.5)
    assert adv[0][0] == pytest.approx(0.5 - 0.0)
    # 'b': önce 'a' düğümü (kendisi hariç = 0.0), sonra 'ab' düğümü (1.0)
    assert adv[0][1] == pytest.approx(1.0 - 0.0)
    # 'd': önce 'ab' düğümü, oradan başka cevap geçmedi -> baseline yok -> 0
    assert adv[0][2] == 0.0


def test_single_answer_group_gets_zero():
    assert token_advantages([["a", "b"]], [1.0]) == [[0.0, 0.0]]
    assert token_advantages([["a", "b"]], [1.0], scale="group_std") == [[0.0, 0.0]]


def test_empty_sequence_is_allowed():
    adv = token_advantages([[], ["a"]], [0.0, 1.0])
    assert adv[0] == []
    assert adv[1][0] == pytest.approx(0.5)


def test_bad_scale_raises():
    with pytest.raises(ValueError):
        token_advantages([["a"]], [1.0], scale="batch")


def test_tree_counts_values_and_render():
    tree, paths = build_tree([[1, 2, 3], [1, 2, 4], [1, 5]], [1.0, 0.0, 0.5])
    assert tree.count[paths[0][1]] == 3
    assert tree.value(paths[0][2]) == pytest.approx(0.5)
    assert tree.value(paths[2][2]) == pytest.approx(0.5)
    text = render_tree(tree)
    assert text.splitlines()[0].startswith("kök  n=3")
    assert "A=" in text
