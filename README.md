# prefix-tree-rl

Önek ağacı (prefix tree) avantajı: critic eğitmeden token başına kredi dağıtan bir RL avantaj tahmincisi.

GRPO bir cevabın bütün token'larına aynı avantajı verir. PPO bunu düzeltmek için ayrı bir critic modeli eğitir. Bu repo üçüncü bir yol öneriyor: aynı prompt'a verilen G cevap zaten bir ağaç oluşturuyor, değer tahmini doğrudan bu ağaçtan okunabilir.

```
V(düğüm) = o önekten geçen cevapların ortalama ödülü
A(token) = V(token'dan sonraki düğüm) − V(token'dan önceki düğüm)
```

## Örnek

`07*42` sorusuna 8 cevap. Model ya hesap makinesini çağırıyor ya da ezberden yazıyor. Ödüller: doğru 1.0, "bilmiyorum" 0.5, yanlış 0, her tool çağrısı −0.02.

```
kök  n=8  V=0.555
├─ A                      n=5  V=0.300  A=-0.255      ← ezbere başlamak: cezalı
│  ├─ 0                   n=3  V=0.333  A=+0.033
│  │  ├─ 2                n=2  V=0.500  A=+0.167
│  │  │  ├─ 94<eos>       n=1  V=1.000  A=+0.500      ← doğru rakam
│  │  │  └─ 84<eos>       n=1  V=0.000  A=-0.500      ← hatanın yapıldığı rakam
│  │  └─ 300<eos>         n=1  V=0.000  A=-0.333
│  ├─ ?<eos>              n=1  V=0.500  A=+0.200
│  └─ 1294<eos>           n=1  V=0.000  A=-0.300
└─ <call>07*42</call>…    n=3  V=0.980  A=+0.425      ← tool kararı: bütün kredi burada
```

- Tool'lu cevaplarda kredinin tamamı `<call>` token'ına gidiyor. Ortak kopyalama kısmı 0 alıyor.
- Şansla doğru bulan ezber cevabında `A` kararı **negatif** (−0.255) alıyor. GRPO bu cevabın bütün token'larına +0.912 verirdi, yani ezberi ödüllendirirdi.
- İkinci senaryoda 7 doğru cevap ve son rakamı yanlış kopyalanmış bir `A0295` var. Sadece `5` token'ı cezalanıyor (−0.875). GRPO −2.475'i `<call>` dahil 13 token'ın hepsine yayardı.

```bash
python examples/calculator_demo.py   # yukarıdaki ağaçları ve tabloları basar
```

## Kullanım

```python
from prefix_tree_rl import token_advantages

adv = token_advantages(
    sequences=[["<call>", "7", "<eos>"], ["A", "9", "<eos>"]],  # token'lar: int, str, herhangi hashable
    rewards=[1.0, 0.0],
    groups=[0, 0],            # aynı prompt'a verilen cevaplar aynı grupta
    scale="none",             # "none": Dr.GRPO ölçeği | "group_std": GRPO ölçeği
    loo=False,                # True: baseline cevabın kendisini dışarıda bırakır
)
# adv[i][k] = i'nci cevabın k'inci token'ının avantajı
```

HF/TRL tensör düzeni (prompt ayrı, completion'lar sağdan pad'li):

```python
from prefix_tree_rl import completion_advantages

adv = completion_advantages(
    completion_ids,     # [N, T]
    completion_mask,    # [N, T], 1 = gerçek token
    rewards,            # [N]
    group_ids,          # [N]
    loss_mask=tool_mask,  # ops.: tool çıktıları gibi kredi almayacak token'lar 0
)                       # -> [N, T]; policy loss'ta advantages.unsqueeze(1) yerine kullanılır
```

## Özellikler (testlerle kanıtlı)

| Özellik | Test |
|---|---|
| Bir cevabın token avantajlarının toplamı **R − grup ortalaması** (Dr.GRPO). Aradaki V'ler teleskop gibi birbirini götürür. Yeni sinyal uydurulmaz, var olan sinyal yeniden dağıtılır. | `test_telescope_*` |
| Cevapların ortak kısmı 0 alır, kredi yolların ayrıldığı token'a gider | `test_credit_goes_to_the_decision…` |
| Şans cezalanır: kötü bir dalda şansla doğru bulan cevabın dal kararı negatif alır | `test_lucky_guess…` |
| Kopyalama hatasında suç sadece yanlış token'a yazılır | `test_copy_error…` |
| `loss_mask` ile ortamın şansı modele yazılmaz: tool farklı sonuç döndürünce ağaç dallanır, ama o dallanmanın kredisi tool token'ında kalır | `test_loss_mask…` |
| `loo=True`: RLOO gibi, baseline'dan cevabın kendisi çıkarılır | `test_loo_*` |
| Gruplar birbirinden bağımsız hesaplanır | `test_groups_are_independent` |

```bash
pip install -e ".[dev]"
python -m pytest -q      # 21 test
```

### Sınırlar

- **Teleskop eşitliğinin koşulu:** her cevabın kendi yaprağı olmalı. Bir cevap diğerinin öneki ise (EOS'suz `a` ve `ab`) kısa olanın toplamı R − ortalama çıkmaz. LLM çıktısında EOS token'ı bunu engeller.
- **Derin düğümlerde tek cevap kalır.** Yollar ayrıldıktan sonra her düğümde n=1 olur, V o cevabın kendi ödülüne eşitlenir ve sonraki token'lar 0 alır. Kredinin tamamı ayrılma noktasına yığılır.
- **Serbest metinde yollar erken ayrılır.** Sıcaklık 1 ile örneklenen uzun cevaplar ilk birkaç token'da birbirinden ayrılır. Kredi o zaman neredeyse rastgele erken token'lara gidebilir. Bu, fikrin gerçek bir LLM'de sınanması gereken asıl zayıf noktası.
- `loss_mask` ile ortam token'ları sıfırlanınca toplam artık R − ortalama olmaz: ortamın şansına düşen kısım bilerek dışarıda bırakılır.

## Şimdiye kadarki kanıt

[voltran](https://github.com/AltanReisoglu/post_train_Detailed/tree/main/voltran) kitindeki oyuncak görevde yapıldı: ~106K parametrelik GPT, hesap makinesi tool'u. Aynı SFT modelinden başlayıp 5 seed ve 40 adım çalıştırıldı:

| Yöntem | 4. adımda pass^4 | İlk 10 adımın ortalaması | Eğri alanı |
|---|---|---|---|
| GRPO | 0.722 | 0.750 | 0.893 |
| önek ağacı, GRPO ölçeği | 0.702 | 0.767 | 0.893 |
| Dr.GRPO | 0.821 | 0.841 | 0.925 |
| **önek ağacı, Dr.GRPO ölçeği** | **0.890** | **0.882** | **0.934** |

- Std'ye bölmeden kullanıldığında önek ağacı ilk 10 adımda **5 seed'in 5'inde** Dr.GRPO'dan hızlı öğreniyor (+0.04).
- Son performansta fark yok: hepsi ~0.96'da buluşuyor.
- GRPO ölçeğinde avantaj kayboluyor ve eğitim daha oynak hale geliyor.

Bu tek bir oyuncak görev. Cevaplar kısa (~13 token) ve karar noktaları az olduğu için genellenebilir bir sonuç değil. Sıradaki adım aynı karşılaştırmayı gerçek bir dil modelinde (Qwen2.5-0.5B-Instruct) yapmak.

## Dosyalar

| Dosya | İçerik |
|---|---|
| [prefix_tree_rl/tree.py](prefix_tree_rl/tree.py) | `PrefixTree`, `token_advantages`, `render_tree`. Saf Python, bağımlılık yok |
| [prefix_tree_rl/torch_ops.py](prefix_tree_rl/torch_ops.py) | HF/TRL düzeni için `completion_advantages`. Kıyaslama için GRPO, Dr.GRPO ve RLOO |
| [examples/calculator_demo.py](examples/calculator_demo.py) | Yukarıdaki iki senaryo |
| [tests/](tests/) | 21 test |
