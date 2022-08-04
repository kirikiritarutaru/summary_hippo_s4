---
marp: true
---

# HiPPOとS4の自分用まとめ

---

#  本スライドの目的
おもしろそうな下記の論文を読み込んで、将来の自分のためにまとめを残す。
- [Gu, Albert, et al. "Hippo: Recurrent memory with optimal polynomial projections." Advances in Neural Information Processing Systems 33 (2020): 1474-1487.](https://proceedings.neurips.cc/paper/2020/file/102f0bb6efb3a6128a3c750dd16729be-Paper.pdf)
- [Gu, Albert, et al. "Combining recurrent, convolutional, and continuous-time models with linear state space layers." Advances in neural information processing systems 34 (2021): 572-585.](https://arxiv.org/pdf/2110.13985.pdf)
- [Gu, Albert, Karan Goel, and Christopher Ré. "Efficiently modeling long sequences with structured state spaces." arXiv preprint arXiv:2111.00396 (2021).](https://arxiv.org/pdf/2111.00396v2.pdf)
---

# HiPPOとは？
一言でいうと
- 長期的な”記憶”を保持しながら、オンラインで時系列データを”要約”する時系列モデリング手法

---
# S4とは？
一言でいうと
- HiPPOのパラメータに適当に制約を課し、計算を高速化した時系列モデリング手法

---

# HiPPOとS4の実装

- https://github.com/HazyResearch/state-spaces

---

# S4の性能まとめ

- https://paperswithcode.com/paper/efficiently-modeling-long-sequences-with-1

---
