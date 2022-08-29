---
marp: true
paginate: true
transition: "none"
math: katex
---

# S4 論文の Appendix まとめ

---

## Sherman-Morrison-Woodburyの公式
任意の行列 $\bm{A},\bm{B},\bm{C}$ に対して次が成り立つ。ここで、$\bm{I}$ は単位行列。
$$
(\bm{A}+\bm{BC})^{-1} = \bm{A}^{-1} - \bm{A}^{-1}\bm{B} (\bm{I}+\bm{CA}^{-1}\bm{B})^{-1}\bm{CA}^{-1}
$$

証明）
実際に計算して確かめる。（太字を省略）
$$
\begin{aligned}
&(A+BC)\left\{ A^{-1}- A^{-1}B(I+CA^{-1}B)^{-1}CA^{-1} \right\} \\
&= I - \textcolor{Red}{\underline{\textcolor{black}{B}}}(I+CA^{-1}B)^{-1}\textcolor{Green}{\underline{\textcolor{black}{CA^{-1}}}} + \textcolor{Red}{\underline{\textcolor{black}{B}}}\textcolor{Green}{\underline{\textcolor{black}{CA^{-1}}}} - \textcolor{Red}{\underline{\textcolor{black}{B}}}CA^{-1}B(I+CA^{-1}B)^{-1}\textcolor{Green}{\underline{\textcolor{black}{CA^{-1}}}} \\
&= I - B\left\{\textcolor{blue}{\underline{\textcolor{black}{(I+CA^{-1}B)^{-1}}}}-I+CA^{-1}B\textcolor{blue}{\underline{\textcolor{black}{(I+CA^{-1}B)^{-1}}}} \right\}CA^{-1} \\
&= I - B\left\{(I+CA^{-1}B)(I+CA^{-1}B)^{-1} - I \right\}CA^{-1} \\
&= I
\end{aligned}
$$

---

## Normal Plus Low-Rank (NPLR)

HiPPO 行列がもつ特殊な構造。
正規行列はユニタリ対角化可能な行列のクラスである。

HiPPO行列 $\bm{A}$ を低ランクの項として記述し、この分解から対角化した行列 $\bm{\Lambda}$ から抽出する。

注意）まだ読んでる途中。記述が支離滅裂になってる。

---

## 定理1
すべてのHiPPO metricsには NPLR 表現をもつ

$$
\bm{A} = \bm{V} \bm{\Lambda} \bm{V}^* - \bm{P}\bm{Q}^\top = \bm{V}(\bm{\Lambda} - (\bm{V}^* \bm{P})(\bm{V}^* \bm{Q})^*)\bm{V}^*
$$

ここで、ユニタリ行列 $\bm{V}\in \mathbb{C}^{N\times N}$、対角行列 $\bm{\Lambda}$、low-rank factorization $\bm{P},\bm{Q} \in \mathbb{R}^{N\times r}$ である。
HiPPO-LegS, LegT, LagT はすべて $r=1$ or $r=2$ をみたす。



