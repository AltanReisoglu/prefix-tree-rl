# prefix-tree-rl

Önek ağacı (prefix tree) avantajı: critic eğitmeden token başına kredi dağıtan bir RL avantaj tahmincisi.

Aynı soruya verilen G cevap, ortak başlangıçları üzerinden bir ağaç oluşturur:

```
V(düğüm) = o önekten geçen cevapların ortalama ödülü
A(token) = V(token'dan sonraki düğüm) − V(token'dan önceki düğüm)
```

- Cevapların ortak kısmı 0 kredi alır. Kredi, yolların ayrıldığı token'a gider.
- Bir cevabın token avantajlarının toplamı R − grup ortalamasına eşittir (Dr.GRPO). Yeni sinyal uydurulmaz, var olan sinyal yeniden dağıtılır.

Bu repo fikri gerçek bir dil modelinde (Qwen2.5-0.5B-Instruct) test ediyor ve GRPO ailesinin diğer yöntemleriyle aynı koşullarda karşılaştırıyor. Algoritmalar [post_train_Detailed/voltran](https://github.com/AltanReisoglu/post_train_Detailed/tree/main/voltran) öğrenme kitinden geliyor.

Hedef donanım: tek bir RTX 4060 Laptop (8 GB).
