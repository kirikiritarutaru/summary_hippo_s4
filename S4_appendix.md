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
実際に計算して確かめる。
$$
\begin{aligned}
&(\bm{A}+\bm{BC})\left\{ \bm{A}^{-1}- \bm{A}^{-1}\bm{B}(\bm{I}+\bm{CA}^{-1}\bm{B})^{-1}\bm{CA}^{-1} \right\} \\
&= \bm{I} - \textcolor{Red}{\underline{\textcolor{black}{\bm{B}}}}(\bm{I}+\bm{CA}^{-1}\bm{B})^{-1}\textcolor{Green}{\underline{\textcolor{black}{\bm{CA}^{-1}}}} + \textcolor{Red}{\underline{\textcolor{black}{\bm{B}}}}\textcolor{Green}{\underline{\textcolor{black}{\bm{CA}^{-1}}}} - \textcolor{Red}{\underline{\textcolor{black}{\bm{B}}}}\bm{CA}^{-1}\bm{B}(\bm{I}+\bm{CA}^{-1}\bm{B})^{-1}\textcolor{Green}{\underline{\textcolor{black}{\bm{CA}^{-1}}}} \\
&= \bm{I} - \bm{B}\left\{\textcolor{blue}{\underline{\textcolor{black}{(\bm{I}+\bm{CA}^{-1}\bm{B})^{-1}}}}-\bm{I}+\bm{CA}^{-1}\bm{B}\textcolor{blue}{\underline{\textcolor{black}{(\bm{I}+\bm{CA}^{-1}\bm{B})^{-1}}}} \right\}\bm{CA}^{-1} \\
&= \bm{I} - \bm{B}\left\{(\bm{I}+\bm{CA}^{-1}\bm{B})(\bm{I}+\bm{CA}^{-1}\bm{B})^{-1} - \bm{I} \right\}\bm{CA}^{-1} \\
&= \bm{I}
\end{aligned}
$$

---

## SSM 畳み込みカーネル

$$
\begin{aligned}
x_0 &=& \overline{\bm{B}}u_0 \qquad x_1 &=& \overline{\bm{AB}}u_0 + \overline{\bm{B}}u_1 \qquad x_2 &=& \overline{\bm{A}}^2 \overline{\bm{B}}u_0 + \overline{\bm{AB}}u_1 + \overline{\bm{B}}u_2 \quad \cdots \\
y_0 &=& \overline{\bm{CB}}u_0 \qquad y_1 &=& \overline{\bm{CAB}}u_0 + \overline{\bm{CB}}u_1 \qquad y_2 &=& \overline{\bm{C}}\overline{\bm{A}}^2 \overline{\bm{B}}u_0 + \overline{\bm{CAB}}u_1 + \overline{\bm{CB}}u_2 \quad \cdots

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



