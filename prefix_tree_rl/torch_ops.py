"""Torch sarmalayıcıları: HF/TRL düzeninde tensörlerle önek ağacı avantajı.

HF/TRL düzeni: her satır bir cevap (completion), prompt ayrı tutulur.

    completion_ids   [N, T]  sağdan pad'lenmiş cevap token'ları
    completion_mask  [N, T]  1 = gerçek token, 0 = padding
    rewards          [N]     her cevabın skaler ödülü
    group_ids        [N]     aynı prompt'a verilen cevaplar aynı grupta

Dönen avantaj [N, T] boyutunda ve token token. GRPO'nun [N] boyutlu avantajının
yerine, policy loss'ta `advantages.unsqueeze(1)` yerine doğrudan kullanılır.

Kıyaslama için sıra-düzeyi tahminciler de burada: grpo, dr_grpo, rloo.
"""

from __future__ import annotations

import torch

from .tree import token_advantages


def _sequences(completion_ids: torch.Tensor, completion_mask: torch.Tensor) -> list[list[int]]:
    ids = completion_ids.detach().cpu().tolist()
    lengths = completion_mask.detach().cpu().long().sum(-1).tolist()
    return [row[:n] for row, n in zip(ids, lengths)]


@torch.no_grad()
def completion_advantages(
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    rewards: torch.Tensor,
    group_ids: torch.Tensor | None = None,
    *,
    loss_mask: torch.Tensor | None = None,
    scale: str = "none",
    loo: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Token başına önek ağacı avantajı, [N, T].

    completion_mask padding'i ayırır: sadece baştaki mask.sum() token ağaca girer.
    loss_mask (ops.) kredi almayacak token'ları sıfırlar, örneğin çok turlu agent'ta
    tool çıktıları. Bu token'lar ağaçta yine yer alır: tool çıktısı değişince yol
    ayrılır, ama o ayrılığın kredisi (ortamın şansı) modele yazılmaz.
    """
    seqs = _sequences(completion_ids, completion_mask)
    groups = None if group_ids is None else group_ids.detach().cpu().tolist()
    per_token = token_advantages(seqs, rewards.detach().float().cpu().tolist(), groups, scale=scale, loo=loo, eps=eps)

    adv = torch.zeros(completion_ids.shape, dtype=torch.float32)
    for i, a in enumerate(per_token):
        if a:
            adv[i, : len(a)] = torch.tensor(a)
    adv = adv.to(completion_ids.device)
    mask = completion_mask if loss_mask is None else completion_mask * loss_mask
    return adv * mask.to(adv.dtype)


# ---------------------------------------------------------------------------
# Sıra-düzeyi tahminciler (kıyaslama için). Hepsi [N] döndürür.
# ---------------------------------------------------------------------------

def _group_stat(rewards: torch.Tensor, group_ids: torch.Tensor, fn) -> torch.Tensor:
    out = torch.empty_like(rewards, dtype=torch.float32)
    for g in torch.unique(group_ids):
        idx = group_ids == g
        out[idx] = fn(rewards[idx].float())
    return out


def dr_grpo_advantages(rewards: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    """R − grup ortalaması (std'ye bölmeden). Önek ağacı toplamının eşiti."""
    return _group_stat(rewards, group_ids, lambda r: r - r.mean())


def grpo_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """(R − grup ortalaması) / (grup std + eps)."""
    return _group_stat(rewards, group_ids, lambda r: (r - r.mean()) / (r.std() + eps) if len(r) > 1 else r * 0)


def rloo_advantages(rewards: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    """R_i − diğerlerinin ortalaması. Dr.GRPO'nun n/(n−1) katı."""
    def f(r):
        n = len(r)
        return (r - (r.sum() - r) / (n - 1)) if n > 1 else r * 0
    return _group_stat(rewards, group_ids, f)


def broadcast(seq_adv: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sıra avantajını [N] -> [N, T] kopyala (GRPO ailesinin yaptığı)."""
    return seq_adv.unsqueeze(-1) * mask.to(seq_adv.dtype)
