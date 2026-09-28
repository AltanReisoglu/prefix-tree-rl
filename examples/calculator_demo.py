"""Önek ağacını hesap makinesi örneğiyle göster (torch gerekmez).

    python examples/calculator_demo.py

Görev: "07*42" sorusu. Model ya hesap makinesi tool'unu çağırır
(<call>07*42</call>, ortam <res>0294</res> döndürür) ya da ezberden cevap yazar.
Ödül: doğru 1.0, "bilmiyorum" (A?) 0.5, yanlış 0, her tool çağrısı −0.02.

Senaryo 1: 8 cevap (3 tool'lu doğru, 1 şanslı ezber, 3 yanlış ezber, 1 "A?").
Senaryo 2: 7 tool'lu doğru + 1 cevapta son rakam yanlış kopyalanmış (A0295).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefix_tree_rl import build_tree, render_tree, token_advantages  # noqa: E402

TOOL_OUTPUT = {"<res>", "</res>"}  # bu blok ortamdan gelir, modele kredi yazılmaz


def tool_answer(final: str) -> list[str]:
    return ["<call>", *"07*42", "</call>", "<res>", *"0294", "</res>", "A", *final, "<eos>"]


def direct_answer(final: str) -> list[str]:
    return ["A", *final, "<eos>"]


def loss_mask(seq: list[str]) -> list[int]:
    """Tool çıktısı (<res> ... </res>) 0, geri kalan 1."""
    mask, inside = [], False
    for tok in seq:
        if tok == "<res>":
            inside = True
        mask.append(0 if inside else 1)
        if tok == "</res>":
            inside = False
    return mask


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def show(title: str, answers: list[tuple[str, list[str], float]]) -> None:
    labels = [a[0] for a in answers]
    seqs = [a[1] for a in answers]
    rewards = [a[2] for a in answers]
    adv = token_advantages(seqs, rewards)
    m, s = mean(rewards), std(rewards)

    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    tree, _ = build_tree(seqs, rewards)
    print(render_tree(tree))

    print("\nCevap başına: token avantajlarının toplamı Dr.GRPO'ya eşit olmalı")
    print(f"  {'cevap':<30}{'R':>6}{'Σ önek':>9}{'Dr.GRPO':>9}{'GRPO':>8}")
    for lab, a, r in zip(labels, adv, rewards):
        print(f"  {lab:<30}{r:>6.2f}{sum(a):>+9.3f}{r - m:>+9.3f}{(r - m) / s:>+8.3f}")

    print("\nKredinin gittiği token'lar (0 olmayanlar; tool çıktısı hariç):")
    for lab, seq, a in zip(labels, seqs, adv):
        mask = loss_mask(seq)
        nz = [f"{tok}:{x:+.3f}" for tok, x, keep in zip(seq, a, mask) if keep and abs(x) > 1e-9]
        print(f"  {lab:<30}{'  '.join(nz) if nz else '(hepsi 0)'}")


def main() -> None:
    show(
        "Senaryo 1: 07*42 sorusuna 8 cevap",
        [
            ("tool, doğru", tool_answer("0294"), 0.98),
            ("tool, doğru", tool_answer("0294"), 0.98),
            ("tool, doğru", tool_answer("0294"), 0.98),
            ("ezber, şans (A0294)", direct_answer("0294"), 1.0),
            ("ezber, yanlış (A0300)", direct_answer("0300"), 0.0),
            ("ezber, yanlış (A0284)", direct_answer("0284"), 0.0),
            ("bilmiyorum (A?)", direct_answer("?"), 0.5),
            ("ezber, yanlış (A1294)", direct_answer("1294"), 0.0),
        ],
    )
    show(
        "Senaryo 2: 7 doğru + 1 kopyalama hatası (A0295)",
        [("tool, doğru", tool_answer("0294"), 0.98)] * 7
        + [("tool, son rakam yanlış (A0295)", tool_answer("0295"), -0.02)],
    )


if __name__ == "__main__":
    main()
