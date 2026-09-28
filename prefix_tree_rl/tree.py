"""Önek ağacı ve token avantajları — saf Python, torch gerektirmez.

Aynı prompt'a verilen G cevap bir ağaç (trie) oluşturur. Her düğüm bir önektir ve iki
sayı tutar: o önekten geçen cevap sayısı (count) ve bu cevapların ödül toplamı (total).

    V(düğüm)  = total / count                     (o önekten devam edince ortalama ödül)
    A(token)  = V(token'dan sonra) − V(token'dan önce)

Bu, critic'in öğrenmeye çalıştığı V'yi grubun kendi örneklerinden okumaktır: ek
rollout gerektirmeyen bir Monte Carlo tahmini.

Teleskop özelliği: bir cevabın token avantajları toplanınca aradaki V'ler birbirini
götürür, geriye V(yaprak) − V(kök) kalır. Yaprak cevabın kendisine aittir, kök grubun
ortalamasıdır. Yani toplam = R_i − ortalama, tam olarak Dr.GRPO'nun sıra avantajı.
Önek ağacı yeni sinyal uydurmaz, var olan sinyali kararın verildiği token'lara dağıtır.

Eşitliğin koşulu: her cevabın yaprağı yalnızca kendisine ait olmalı. İki istisna:
  * Bir cevap diğerinin öneki ise ("a" ve "ab"), kısa olanın yaprağı uzun olanın ara
    düğümüdür; toplam V(o düğüm) − ortalama olur. LLM çıktısında her tamamlanan cevap EOS
    ile bittiği için bu olmaz. EOS'suz dizilerle çalışıyorsan sonlarına bir bitiş token'ı ekle.
  * İki cevap token token aynı ama ödülleri farklıysa (stokastik ödül), yaprak ikisinin
    ortalamasını tutar.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field

SCALES = ("none", "group_std")


@dataclass
class PrefixTree:
    """Tek bir grubun ağacı. Düğüm 0 kök, yani boş önek (ya da ortak prompt)."""

    children: dict[tuple[int, Hashable], int] = field(default_factory=dict)  # (düğüm, token) -> çocuk
    count: list[int] = field(default_factory=lambda: [0])
    total: list[float] = field(default_factory=lambda: [0.0])
    token: list[Hashable] = field(default_factory=lambda: [None])
    parent: list[int] = field(default_factory=lambda: [-1])

    def add(self, seq: Sequence[Hashable], reward: float) -> list[int]:
        """Bir cevabı ekle; kökten yaprağa düğüm yolunu döndür (uzunluk len(seq) + 1).

        path[k] = ilk k token yazıldıktan sonraki düğüm. Token k'nin öncesi path[k],
        sonrası path[k + 1].
        """
        node, path = 0, [0]
        for tok in seq:
            key = (node, tok)
            child = self.children.get(key)
            if child is None:
                child = len(self.count)
                self.children[key] = child
                self.count.append(0)
                self.total.append(0.0)
                self.token.append(tok)
                self.parent.append(node)
            node = child
            path.append(node)
        for n in path:
            self.count[n] += 1
            self.total[n] += reward
        return path

    def value(self, node: int) -> float:
        return self.total[node] / self.count[node]

    def value_without(self, node: int, reward: float) -> float | None:
        """Bir cevabın kendisi hariç ortalama (leave-one-out). Düğümde başka cevap yoksa None."""
        if self.count[node] <= 1:
            return None
        return (self.total[node] - reward) / (self.count[node] - 1)

    def kids(self, node: int) -> list[int]:
        return [c for (p, _), c in self.children.items() if p == node]

    @property
    def num_nodes(self) -> int:
        return len(self.count)


def _std(xs: list[float]) -> float:
    """Örneklem standart sapması (n − 1), torch.std ile aynı."""
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def token_advantages(
    sequences: Sequence[Sequence[Hashable]],
    rewards: Sequence[float],
    groups: Sequence[Hashable] | None = None,
    *,
    scale: str = "none",
    loo: bool = False,
    eps: float = 1e-6,
) -> list[list[float]]:
    """Her cevabın her token'ı için önek ağacı avantajı.

    sequences  cevaplar (token listeleri). Aynı gruptakiler aynı prompt'a verilmiş olmalı;
               prompt'u dahil etmek gerekmez (ortak olduğu için avantajı zaten 0 olur).
    rewards    her cevabın skaler ödülü
    groups     her cevabın grup kimliği; None ise hepsi tek grup
    scale      "none"      -> ham farklar (Dr.GRPO ölçeği; toplam = R − ortalama)
               "group_std" -> grubun ödül std'sine böl (GRPO ölçeği)
    loo        True -> baseline cevabın kendisini dışarıda bırakır (RLOO gibi). O önekte
               başka cevap yoksa (count = 1) karşılaştıracak kimse olmadığı için avantaj 0.
               Bedeli: teleskop toplamı artık tam R − ortalama değil.

    Dönüş: out[i][k] = sequences[i]'nin k'inci token'ının avantajı.
    """
    if scale not in SCALES:
        raise ValueError(f"bilinmeyen scale: {scale!r}, seçenekler: {SCALES}")
    if len(sequences) != len(rewards):
        raise ValueError("sequences ve rewards aynı uzunlukta olmalı")
    if groups is None:
        groups = [0] * len(sequences)

    members: dict[Hashable, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)

    out: list[list[float]] = [[] for _ in sequences]
    for idx in members.values():
        tree = PrefixTree()
        paths = {i: tree.add(sequences[i], float(rewards[i])) for i in idx}
        denom = 1.0
        if scale == "group_std":
            denom = _std([float(rewards[i]) for i in idx]) + eps

        for i in idx:
            path, r_i = paths[i], float(rewards[i])
            adv = []
            for k in range(len(path) - 1):
                before, after = path[k], path[k + 1]
                if loo:
                    base = tree.value_without(before, r_i)
                    a = 0.0 if base is None else tree.value(after) - base
                else:
                    a = tree.value(after) - tree.value(before)
                adv.append(a / denom)
            out[i] = adv
    return out


def build_tree(sequences: Sequence[Sequence[Hashable]], rewards: Sequence[float]) -> tuple[PrefixTree, list[list[int]]]:
    """Tek grup için ağacı kur; (ağaç, her cevabın düğüm yolu)."""
    tree = PrefixTree()
    paths = [tree.add(s, float(r)) for s, r in zip(sequences, rewards)]
    return tree, paths


def render_tree(tree: PrefixTree, show=str, root: int = 0) -> str:
    """Ağacı metin olarak çiz. Aynı cevapların geçtiği tek çocuklu zincirler tek satıra sıkışır.

        kök  n=8  V=0.555
        ├─ <call>07*42…   n=3  V=0.980  A=+0.425
        └─ A              n=5  V=0.300  A=−0.255
    """
    lines = [f"kök  n={tree.count[root]}  V={tree.value(root):.3f}"]

    def ordered_kids(n: int) -> list[int]:
        return sorted(tree.kids(n), key=lambda c: -tree.count[c])

    def chain(n: int) -> tuple[str, int]:
        label, cur = [show(tree.token[n])], n
        while True:
            ks = ordered_kids(cur)
            if len(ks) != 1 or tree.count[ks[0]] != tree.count[cur]:
                return "".join(label), cur
            cur = ks[0]
            label.append(show(tree.token[cur]))

    def walk(n: int, indent: str) -> None:
        ks = ordered_kids(n)
        for j, c in enumerate(ks):
            last = j == len(ks) - 1
            label, end = chain(c)
            a = tree.value(c) - tree.value(n)
            lines.append(
                f"{indent}{'└─' if last else '├─'} {label:<22} n={tree.count[c]}  V={tree.value(c):.3f}  A={a:+.3f}"
            )
            walk(end, indent + ("   " if last else "│  "))

    walk(root, "")
    return "\n".join(lines)
