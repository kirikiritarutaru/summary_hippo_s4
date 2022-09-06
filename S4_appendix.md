---
marp: true
paginate: true
transition: "none"
math: katex
---

<style>
blockquote {
    border-top: 0.1em dashed #555;
    font-size: 60%;
    margin-top: auto;
}
</style>


# S4 論文の個人的補足と Appendix まとめ

---

## Sherman-Morrison-Woodburyの公式（逆行列補題）
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

> 参考：https://tochikuji.hatenablog.jp/entry/20130813/1376371669
> 逆行列補題とカルマンフィルタの関係 https://cookie-box.hatenablog.com/entry/2018/12/14/023016

---

## ヴァンデルモンドの行列式
$n$ を2以上の自然数とし、実数 $x_1, \dots, x_n$ に対し、
$$
V =
\begin{pmatrix}
1 & x_1 & \cdots & x_1^{n-1}\\
1 & x_2 & \cdots & x_2^{n-1}\\
\vdots & \vdots & \ddots & \vdots \\
1 & x_n & \cdots & x_n^{n-1}\\
\end{pmatrix}
$$
とおく。このとき、行列式 $\det V$ を**ヴァンデルモンドの行列式**とよび、
$$
\det V = \prod_{1\le i\le j \le n} (x_j - x_i)
$$
となる。$x_1,\dots,x_n$ が互いに異なるとき、$V$ の逆行列が存在する。

> 参考：長谷川線形代数[改訂版] p.209 や機械学習のための関数解析入門 p.119

---

## ヴァンデルモンドの行列式の応用
$x_1, \dots, x_n$ を互いに異なる実数とする。$x_1, \dots, x_n$ 上で定義された関数 $f$ に対し、
$$
f(x_j) = p(x_j) \quad (j=1,\dots, n)
$$
をみたす $n-1$ 次以下の多項式 $p(x)=\sum^{n-1}_{j=0}c_j x^j$ が存在する。これは、
$$
\begin{pmatrix}
1 & x_1 & \cdots & x_1^{n-1}\\
1 & x_2 & \cdots & x_2^{n-1}\\
\vdots & \vdots & \ddots & \vdots \\
1 & x_n & \cdots & x_n^{n-1}\\
\end{pmatrix}

\begin{pmatrix}
c_0 \\
c_1 \\
\vdots \\
c_{n-1}
\end{pmatrix}
=
\begin{pmatrix}
f(x_1) \\
f(x_2) \\
\vdots \\
f(x_n)
\end{pmatrix}
$$
をみたす $c_0, \dots, c_{n-1}$ を求めることと同じ。
両辺に左から $V^{-1}$ をかけることで解ける。

> 参考：長谷川線形代数[改訂版] p.209 や機械学習のための関数解析入門 p.119

---

## 畳み込み（合成積）
関数（数列）を並行移動しながら、もう一方の関数（数列）に重ね足し合わせる演算

連続の場合： 関数 $f(x),\ g(x)$ から新しい関数 $h(x)$ を作る
$$
h(x) = \int^\infty_{-\infty} f(t) g(x-t)dt
$$

離散の場合： 数列 $\{a_n\}, \{b_n\}$ から新しい数列 $\{c_n\}$ を作る
$$
c_n= \sum^n_{t=0} a_tb_{n-t}
$$

> 参考：https://manabitimes.jp/math/954

---

## 畳み込み定理
フーリエ変換を $\mathcal{F}(x)$ とし、関数 $f,g$ の畳み込みを $f*g$ とすると、
$$
\mathcal{F}(f*g) = \mathcal{F}(f)\mathcal{F}(g)
$$

証明）
実際に計算して確かめる。
$$
\begin{aligned}
\mathcal{F}\left(f(t)*g(t) \right) &= \int^\infty_{-\infty} e^{-i\omega t} \left[\int^\infty_{-\infty} f(x)g(t-x)dx \right] dt \\
&= \int^\infty_{-\infty} f(x) \left[\int^\infty_{-\infty}e^{-i\omega t}g(t-x) dx \right] dt \\
&= \int^\infty_{-\infty}f(x)e^{-i\omega t} \left[e^{-i \omega (t-x)}g(t-x)dt \right] dx \\
&= \mathcal{F}(f(t))\mathcal{F}(g(t))
\end{aligned}
$$

TODO: 文字の定義を書け

---

## 畳み込み定理

畳み込みの計算は、高速フーリエ変換 (FFT) を用いて高速に計算できる。
$N$ 点の信号 $h_n$ と $x_n$ の畳み込み $y_n$ を計算する場合、
- 畳み込みを直接計算（計算量：$O\left(N^2 \right)$）
線形畳み込み： $y_n = \sum^{N-1}_{k=0} h_k x_{n-k} \quad (n=0,1,2,\dots, 2N-2)$
- FFT $\mathcal{F}$ →積をとる→ iFFT $\mathcal{F}^{-1}$（計算量：$O\left(N\log N \right)$）
循環畳み込み： $y = \mathcal{F}^{-1}(\mathcal{F}(h)\mathcal{F}(x))$

$N_h$ 点の信号 $h_n$ と $N_x$ 点の信号 $x_n$ に、$N_{x-1}$ 点と $N_h-1$ 点のゼロづめを行い、$N_h+N_x-1$ 点の循環畳み込みを行うことで、線形畳み込みが実現できる。

> 参考：Python対応ディジタル信号処理 p.64~70

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

注意）まだ読んでる途中。記述が支離滅裂。

---

## 定理1
すべてのHiPPO metricsは NPLR 表現をもつ

$$
\bm{A} = \bm{V} \bm{\Lambda} \bm{V}^* - \bm{P}\bm{Q}^\top = \bm{V}(\bm{\Lambda} - (\bm{V}^* \bm{P})(\bm{V}^* \bm{Q})^*)\bm{V}^*
$$

ここで、ユニタリ行列 $\bm{V}\in \mathbb{C}^{N\times N}$、対角行列 $\bm{\Lambda}$、low-rank factorization $\bm{P},\bm{Q} \in \mathbb{R}^{N\times r}$ である。
HiPPO-LegS, LegT, LagT はすべて $r=1$ or $r=2$ をみたす。

---

## Diagonal Plus Low-Rank (DPLR)

NPLR にさらに制限を加える。
行列 $\bm{A}\in \mathbb{R}^{n\times n}$ が

$$
\bm{A} = \bm{\Lambda} - pq^\top
$$
と書けるとき、$\bm{A}$ はDPLR表現を持つという。
ここで、$\bm{\Lambda}$ は対角行列、$p,q \in \mathbb{R}^{n\times k} \ (k\ll n)$ である。

---

## 定理2 S4 Recurrence
ステップサイズ $\Delta$ が与えられたとき、状態数を $N$ とすると、離散SSMは1ステップ $O(N)$ で計算できる。

離散SSMの式（状態空間モデルの離散化のスライドに記載 ）
$$
\begin{aligned}
\bm{\overline{A}} &= \left(\bm{I} - \frac{\Delta}{2}\cdot \bm{A} \right)^{-1} \left(\bm{I} + \frac{\Delta}{2}\cdot \bm{A} \right) \\
\bm{\overline{B}} &= \left(\bm{I} - \frac{\Delta}{2}\cdot \bm{A} \right)^{-1}\Delta \bm{B},\quad \bm{\overline{C}} = \bm{C}
\end{aligned}
$$


