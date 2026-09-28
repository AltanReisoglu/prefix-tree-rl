"""Torch sarmalayıcılarının testleri (HF/TRL tensör düzeni)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefix_tree_rl import (  # noqa: E402
    completion_advantages,
    dr_grpo_advantages,
    grpo_advantages,
    rloo_advantages,
    token_advantages,
)

PAD = 0


def _batch(seqs):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), PAD, dtype=torch.long)
    mask = torch.zeros(len(seqs), T)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids, mask


SEQS = [[5, 6, 7], [5, 6, 8, 9], [5, 10], [11], [5, 6, 7]]
REWARDS = torch.tensor([1.0, 0.0, 0.5, 0.25, 1.0])


def test_matches_pure_python_and_padding_is_zero():
    ids, mask = _batch(SEQS)
    adv = completion_advantages(ids, mask, REWARDS)
    ref = token_advantages(SEQS, REWARDS.tolist())
    for i, s in enumerate(SEQS):
        assert adv[i, : len(s)].tolist() == pytest.approx(ref[i], abs=1e-6)
        assert torch.all(adv[i, len(s):] == 0)


def test_row_sum_equals_dr_grpo():
    """Farklı diziler + aynı dizilerin aynı ödülü -> satır toplamı Dr.GRPO."""
    ids, mask = _batch(SEQS)
    groups = torch.zeros(len(SEQS), dtype=torch.long)
    adv = completion_advantages(ids, mask, REWARDS, groups)
    assert torch.allclose(adv.sum(-1), dr_grpo_advantages(REWARDS, groups), atol=1e-6)


def test_pad_token_inside_sequence_is_kept():
    """Uzunluğu mask belirler: dizinin içindeki PAD id'li token da ağaca girer."""
    seqs = [[5, PAD, 7], [5, PAD, 8]]
    ids, mask = _batch(seqs)
    adv = completion_advantages(ids, mask, torch.tensor([1.0, 0.0]))
    assert adv[0, 2].item() == pytest.approx(0.5)
    assert adv[1, 2].item() == pytest.approx(-0.5)


def test_loss_mask_zeroes_tool_tokens_but_tree_still_branches():
    """Tool çıktısı (loss_mask 0) kredi almaz; farklı çıktı yine ayrı dal açar."""
    seqs = [[1, 50, 2], [1, 60, 2]]  # 50/60: ortamın döndürdüğü farklı sonuç
    ids, mask = _batch(seqs)
    lm = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    adv = completion_advantages(ids, mask, torch.tensor([1.0, 0.0]), loss_mask=lm)
    assert adv[:, 1].abs().sum() == 0  # ortamın şansı modele yazılmadı
    assert adv[:, 2].abs().sum() == 0  # ortam dalından sonra model aynı şeyi yazdı: 0
    assert adv[:, 0].abs().sum() == 0  # ortak ilk token: 0


def test_groups_and_device_roundtrip():
    ids, mask = _batch(SEQS + SEQS)
    r = torch.cat([REWARDS, 1 - REWARDS])
    g = torch.tensor([0] * 5 + [1] * 5)
    adv = completion_advantages(ids, mask, r, g)
    alone = completion_advantages(*_batch(SEQS), REWARDS)
    assert torch.allclose(adv[:5], alone, atol=1e-6)
    assert adv.device == ids.device


def test_sequence_baselines():
    r = torch.tensor([1.0, 0.0, 0.5, 0.5])
    g = torch.zeros(4, dtype=torch.long)
    dr = dr_grpo_advantages(r, g)
    assert torch.allclose(dr, r - r.mean())
    assert torch.allclose(grpo_advantages(r, g), (r - r.mean()) / (r.std() + 1e-6))
    assert torch.allclose(rloo_advantages(r, g), dr * 4 / 3, atol=1e-6)  # RLOO = n/(n-1) · Dr.GRPO
