---
marp: true
paginate: true
transition: "none"
math: katex
---

# S4 論文の Appendix まとめ

---

## Sherman-Morrison-Woodburyの公式
任意の行列 $A,B,C$ に対して次が成り立つ。ここで、$I$ は単位行列。
$$
(A+BC)^{-1} = A^{-1} - A^{-1}B (I+CA^{-1}B)^{-1}CA^{-1}
$$

証明）
実際に計算して確かめる。
$$
\begin{aligned}
&(A+BC)\left\{ A^{-1}- A^{-1}B(I+CA^{-1}B)^{-1}CA^{-1} \right\} \\
&= I - \textcolor{Red}{\underline{\textcolor{black}{B}}}(I+CA^{-1}B)^{-1}\textcolor{Green}{\underline{\textcolor{black}{CA^{-1}}}} + \textcolor{Red}{\underline{\textcolor{black}{B}}}\textcolor{Green}{\underline{\textcolor{black}{CA^{-1}}}} - \textcolor{Red}{\underline{\textcolor{black}{B}}}CA^{-1}B(I+CA^{-1}B)^{-1}\textcolor{Green}{\underline{\textcolor{black}{CA^{-1}}}} \\
&= I - B\left\{\textcolor{blue}{\underline{\textcolor{black}{(I+CA^{-1}B)^{-1}}}}-I+CA^{-1}B\textcolor{blue}{\underline{\textcolor{black}{(I+CA^{-1}B)^{-1}}}} \right\}CA^{-1} \\
&= I - B\left\{(I+CA^{-1}B)(I+CA^{-1}B)^{-1} - I \right\}CA^{-1} \\
&= I
\end{aligned}
$$

