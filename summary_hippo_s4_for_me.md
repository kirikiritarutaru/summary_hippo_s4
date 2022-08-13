---
marp: true
paginate: true
transition: "none"
math: katex
---

# HiPPOとS4の自分用まとめ

---

##  本スライドの目的
おもしろそうな下記の論文を読み込んで、将来の自分のためにまとめを残す。
- [Gu, Albert, et al. "Hippo: Recurrent memory with optimal polynomial projections." Advances in Neural Information Processing Systems 33 (2020): 1474-1487.](https://proceedings.neurips.cc/paper/2020/file/102f0bb6efb3a6128a3c750dd16729be-Paper.pdf)
- [Gu, Albert, et al. "Combining recurrent, convolutional, and continuous-time models with linear state space layers." Advances in neural information processing systems 34 (2021): 572-585.](https://arxiv.org/pdf/2110.13985.pdf)
- [Gu, Albert, Karan Goel, and Christopher Ré. "Efficiently modeling long sequences with structured state spaces." arXiv preprint arXiv:2111.00396 (2021).](https://arxiv.org/pdf/2111.00396v2.pdf)

---

## HiPPO とは？

一言でいうと、
- 長期的な”記憶”を保持しながら、オンラインで時系列データを”要約”するフレームワーク

---

## 背景
- **時系列データ**のモデリングと学習はいろんなところで使われる基本的な問題
  - 言語モデリング、音声認識、映像処理、強化学習

#### 時系列データをモデリング/学習する上での中心的な課題は？
- 多くのデータが処理されるにつれてどんどん増える過去のデータを**記憶**すること

#### 記憶するには、何が必要？
- 過去のデータを取り扱いやすい表現に変えること
  - 蓄積される過去のデータ全体を、有限のメモリで表す方法を獲得しなければならない
- 次々に入ってくるデータから、逐次的に「過去のデータの表現」を更新できなければならない

---

## 既存手法
- 過去の情報を取り込みながら、時間と共に変化していく状態をモデル化するアプローチ
    - RNN, LSTM, GRU
    - 最近の論文だと、[フーリエ再帰ユニット](https://arxiv.org/pdf/1803.06585.pdf)、[ルジャンドル再帰ユニット](https://proceedings.neurips.cc/paper/2019/file/952285b9b7e7a1be5aa7849f32ffff05-Paper.pdf)

### 既存手法の課題
1. 数万ステップもあるようなデータだと記憶を保持できない
2. 入力データの時間スケールやシーケンス長に対して仮定をおいている
3. 「長期依存性をどの程度うまくとらえられるか」についての理論的保証がない

---

## HiPPOが解決したこと
1. 数百万ステップのデータでも記憶を保持できる（表現の獲得）
2. 時間スケールへの事前の仮定がなく、任意のシーケンス長の入力データを扱える
3. 記憶を扱う方法を理論的に定式化し、厳密な理論的保証を与える

---

## S4とは？
一言でいうと
- HiPPOのパラメータに制約を課し、計算高速化

---

## HiPPOとS4の実装

- https://github.com/HazyResearch/state-spaces

---

## S4の性能まとめ

- https://paperswithcode.com/paper/efficiently-modeling-long-sequences-with-1

---

## 付録

---

## ルジャンドル多項式
区間$[-1,1]$の$n$次のルジャンドル多項式の定義（ロドリゲスの公式）：

$P_n(x) = \frac{(-1)^n}{2^n n!} \frac{d^n}{dx^n}(1-x^2)^n \quad (n=0,1,2,\dots)$

具体的に計算すると、
$$
\begin{aligned}
P_0(x) &= 1 \\
P_1(x) &= x \\
P_2(x) &= \frac{3}{2}x^2 - \frac{1}{2} \\
P_3(x) &= \frac{5}{2}x^3 - \frac{3}{2}x
\end{aligned}
$$

---

## ルジャンドル多項式の性質
1. 正規化したルジャンドル多項式 $\sqrt{\frac{2n+1}{2}}P_n(x)$ は、区間 $[-1,1]$ 上の $L_2$ 空間の完全正規直交系
2. $n$次のルジャンドル多項式は、$n-1$次以下のべき関数と直交する
  $\int^{1}_{-1}x^m P_n(x)dx=0 \quad (m<n)$

3. $P_n(1)=1,\ P_n(-1)=(-1)^n$

4. ボネの漸化式 （証明は高木解析概論P.130参照）
  $(n+1)P_{n+1}(x) - (2n+1)xP_n(x)+nP_{n-1}(x) = 0 \quad (n\ge 1)$

---

## ルジャンドル多項式の性質

5. HiPPOの論文でつかう漸化式

$(2n+1)P_n(x)=P^\prime_{n+1}(x)-P^\prime_{n-1}(x)$

$P^\prime_{n+1}(x)-xP^\prime_n(x)=(n+1)P_n(x)$

$P^\prime_n(x) = (2n-1)P_{n-1}(x) + (2n-3)P_{n-2}(x) + \cdots$

$(x+1)P^\prime_n(x) =nP_n(x) + (2n-1)P_{n-1}(x) + (2n-3)P_{n-2}(x) + \cdots$

---

## HiPPO-LegT の導出 (論文Appendix D.1)
### 測度と基底
- 測度：”過去”の重み付け
$\omega(t,x) = \frac{1}{\theta} \mathbb{I}_{[t-\theta,t]}$
- 基底：正規化されたルジャンドル多項式
$p_n(t,x) = \sqrt{2n+1} \ P_n(\frac{2(x-t)}{\theta} + 1)$
- 入力信号を近似する関数
$g(t,x) = \sum g_n(t,x) = \sum \lambda_n p_n(t,x)$
  - $x=t$ の時の $g_n$ は、 （←ルジャンドル多項式の性質3.より）
  $g_n(t,t)= \lambda_n \sqrt{2n+1}$
  - $x=t-\theta$ の時の $g_n$ は、
  $g_n(t,t-\theta)= \lambda_n (-1)^n \sqrt{2n+1}$


---

## HiPPO-LegT の導出 (論文Appendix D.1)
### 時間 $t$ に関する微分
- 測度の微分（ここで、 $\delta$ はディラックのデルタ）
$\frac{\partial}{\partial t} \omega(t,x) = \frac{1}{\theta} \delta_t - \frac{1}{\theta} \delta_{t-\theta}$
- 近似する関数 $g_n$ の微分（ルジャンドル多項式の性質5.より）
$$
\begin{aligned}
\frac{\partial}{\partial t} g_n(t,x) &= \lambda_n \sqrt{2n+1} \cdot \frac{-2}{\theta} P_n^\prime \left( \frac{2(x-t)}{\theta} +1 \right) \\
&= \lambda_n \sqrt{2n+1} \cdot \frac{-2}{\theta} \left[ (2n-1)P_{n-1} \left( \frac{2(x-t)}{\theta}+1 \right) + (2n-5) P_{n-3}\left( \frac{2(x-t)}{\theta} +1 \right)+ \cdots\right] \\
&= -\lambda_n \sqrt{2n+1}\cdot \frac{2}{\theta} \left[ \lambda^{-1}_{n-1} \sqrt{2n+1} \cdot g_{n-1}(t,x) + \lambda_{n-3}^{-1} \sqrt{2n-3} \cdot g_{n-3}(t,x)+\cdots \right]
\end{aligned}
$$

---

## HiPPO-LegT の導出 (論文Appendix D.1)
### 幅 $\theta$ のスライディングウィンドウを動かしながらの関数近似
後ほど、 $\frac{d}{dt}c(t)$ を調べる時に、$f(t-\theta)$ の値が必要なので準備
$$
\begin{aligned}
f_{\le t}(x) &\approx g(t,x) &=& \sum^{N-1}_{k=0} \lambda^{-1}_k c_k(t) \sqrt{2k+1} \cdot P_k \left( \frac{2(x-t)}{\theta} +1 \right) \\
f(t-\theta) &\approx g(t-\theta) &=& \sum^{N-1}_{k=0} \lambda_k^{-1} c_k(t) \sqrt{2k+1} \cdot (-1)^k
\end{aligned}
$$

---

## HiPPO-LegT の導出 (論文Appendix D.1)
### 係数のダイナミクス1
「時間 $t$ に関する微分」に記載の式を用い、
$$
\begin{aligned}
\frac{d}{dt} c_n(t) =& \int f(x) \left( \frac{\partial}{\partial t} g_n (t,x) \right) \omega (t,x) dx \\ &+ \int f(x) g_n (t,x) \left( \frac{\partial}{\partial t} \omega (t,x) \right) \\

 =& -\lambda_n \sqrt{2n+1} \cdot \frac{2}{\theta} \left[ \lambda^{-1}_{n-1} \sqrt{2n-1}\cdot c_{n-1}(t) + \lambda_{n-3}^{-1} \sqrt{2n-5} \cdot c_{n-3}(t) + \cdots \right] \\
 &+ \frac{1}{\theta}f(t) g_n (t,t) -\frac{1}{\theta} f(t-\theta)g_n (t,t-\theta)

\end{aligned}
$$

---

## HiPPO-LegT の導出 (論文Appendix D.1)
### 係数のダイナミクス2
「測度と基底」と「幅 $\theta$ のスライディングウィンドウを動かしながらの関数近似」に記載の式を用い、
$$
\begin{aligned}
\frac{d}{dt} c_n(t) \approx& -\frac{\lambda_n}{\theta} \sqrt{2n+1} \cdot 2 \left[  \sqrt{2n-1}\cdot \frac{c_{n-1}(t)}{\lambda_{n-1}} + \sqrt{2n-5} \cdot \frac{c_{n-3}(t)}{\lambda_{n-3}} + \cdots \right] \\
 &+ \sqrt{2n+1} \cdot \frac{\lambda_n}{\theta} f(t) - \sqrt{2n+1} \cdot \frac{\lambda_n}{\theta} (-1)^n \sum^{N-1}_{k=0} \sqrt{2k+1} \cdot \frac{c_k(t)}{\lambda_k} (-1)^k
\end{aligned}
$$