---
marp: true
paginate: true
transition: "none"
math: katex
---

# HiPPO 論文の Appendix まとめ

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
1. 正規化したルジャンドル多項式 $\sqrt{\frac{2n+1}{2}}P_n(x)$ は、区間 $[-1,1]$ 上の $L_2$ 空間の完全正規直交関数系
2. $n$次のルジャンドル多項式は、$n-1$次以下のべき関数と直交する
  $\int^{1}_{-1}x^m P_n(x)dx=0 \quad (m<n)$

3. $P_n(1)=1,\ P_n(-1)=(-1)^n$

4. ボネの漸化式 （証明は高木解析概論 p.130 参照）
  $(n+1)P_{n+1}(x) - (2n+1)xP_n(x)+nP_{n-1}(x) = 0 \quad (n\ge 1)$

---

## ルジャンドル多項式の性質
5. HiPPOの論文でつかう漸化式
- $(2n+1)P_n(x)=P^\prime_{n+1}(x)-P^\prime_{n-1}(x)$
- $P^\prime_{n+1}(x)-xP^\prime_n(x)=(n+1)P_n(x)$
- $P^\prime_n(x) = (2n-1)P_{n-1}(x) + (2n-3)P_{n-2}(x) + \cdots$
- $(x+1)P^\prime_n(x) =nP_n(x) + (2n-1)P_{n-1}(x) + (2n-3)P_{n-2}(x) + \cdots$

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
- 測度の微分（ここで、 $\delta$ はクロネッカーのデルタ）
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
### 係数のダイナミクス 1
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
### 係数のダイナミクス 2
「測度と基底」と「幅 $\theta$ のスライディングウィンドウを動かしながらの関数近似」に記載の式を用い、
$$
\begin{aligned}
\text{（続き）} \approx& -\frac{\lambda_n}{\theta} \sqrt{2n+1} \cdot 2 \left[  \sqrt{2n-1}\cdot \frac{c_{n-1}(t)}{\lambda_{n-1}} + \sqrt{2n-5} \cdot \frac{c_{n-3}(t)}{\lambda_{n-3}} + \cdots \right] \\
 &+ \sqrt{2n+1} \cdot \frac{\lambda_n}{\theta} f(t) - \sqrt{2n+1} \cdot \frac{\lambda_n}{\theta} (-1)^n \sum^{N-1}_{k=0} \sqrt{2k+1} \cdot \frac{c_k(t)}{\lambda_k} (-1)^k
\end{aligned}
$$

---

## HiPPO-LegT の導出 (論文Appendix D.1)
### 係数のダイナミクス 3
$\sqrt{2n+1}\cdot \frac{\lambda_n}{\theta} f(t)$ 以外の項をまとめて、
$$
\begin{aligned}
\text{（続き）} =& -\frac{\lambda_n}{\theta} \sqrt{2n+1} \cdot \sum^{N-1}_{k=0} M_{nk} \sqrt{2k+1} \cdot \frac{c_k(t)}{\lambda_k} + \sqrt{2n+1}\cdot \frac{\lambda_n}{\theta} f(t)
\end{aligned}
$$
ここで、
$$
M_{nk} =
\begin{cases}
1 &\text{if}& k\le n \\
(-1)^{n-k} &\text{if}& k\ge n
\end{cases}
$$

---

## HiPPO-LegT の導出 (論文Appendix D.1)
### 係数のダイナミクス 4
linear ODE $\frac{d}{dt} c(t) = -\frac{1}{\theta} A c(t) + \frac{1}{\theta}Bf(t)$ の形に書き直すと、

$\lambda_n=1$ の場合（＝正規化したルジャンドル多項式を使う場合）、
$$
A_{nk} = \sqrt{2n+1} \cdot \sqrt{2k+1} \cdot
\begin{cases}
1 &\text{if}& k\le n \\
(-1)^{n-k} &\text{if}& k\ge n
\end{cases} \\
B_n=\sqrt{2n+1}
$$
$\lambda_n=\sqrt{2n+1}\cdot (-1)^n$ の場合（＝ルジャンドル多項式をそのまま使う場合）、
$$
A_{nk} = (2n+1) \cdot
\begin{cases}
(-1)^{n-k} &\text{if}& k\le n \\
1 &\text{if}& k\ge n
\end{cases} \\
B_n=(2n+1) (-1)^n
$$

---

## HiPPO-LegT の導出 (論文Appendix D.1)
### 得た係数から入力信号 $f$ を近似する関数を再構成する時

任意の時間 $t$ において、
$$
f(x) \approx g^{(t)}(x) = \sum_n \lambda_n^{-1} c_n(t) \sqrt{2n+1} \cdot P_n \left( \frac{2(x-t)}{\theta} +1 \right)
$$


---

## HiPPO-LegS の導出 (論文Appendix D.3)
HiPPO-LegT と同様の流れで導出する。HiPPO-LegS が本命である。

### 測度と基底
- 測度：”過去”の重み付け
$\omega(t,x) = \frac{1}{t} \mathbb{I}_{[0,t]}$
- 基底：正規化されたルジャンドル多項式
$p_n(t,x) = \sqrt{2n+1} \ P_n(\frac{2x}{t} - 1)$
- 入力信号を近似する関数 （ここで、$\lambda_n=1$ とする）
$g(t,x) = \sum g_n(t,x) = \sum p_n(t,x)$

---

## HiPPO-LegS の導出 (論文Appendix D.3)
### 時間 $t$ に関する微分 1
- 測度の微分（ここで、 $\delta$ はクロネッカーのデルタ）
$\frac{\partial}{\partial t} \omega(t,\cdot) = -t^{-2}\mathbb{I}_{[0,t]} + t^{-1}\delta_t = t^{-1} \left(-\omega(t) + \delta_t \right)$
- 近似する関数 $g_n$ の微分
$$
\begin{aligned}
\frac{\partial}{\partial t} g_n(t,x) &= -\sqrt{2n+1} \cdot 2xt^{-2} P_n^\prime \left( \frac{2x}{t} -1 \right) \\
&= -\sqrt{2n+1} \cdot t^{-1} \left(\frac{2x}{t} -1 +1 \right) P_n^\prime \left( \frac{2x}{t} -1 \right)
\end{aligned}
$$

---

## HiPPO-LegS の導出 (論文Appendix D.3)
### 時間 $t$ に関する微分 2
- 近似する関数 $g_n$ の微分（ルジャンドル多項式の性質5.より）
見た目を整えるため $z=\frac{2x}{t} -1$ とおくと、
$$
\begin{aligned}
\frac{\partial}{\partial t} g_n(t,x) &= -\sqrt{2n+1} \cdot t^{-1} \left(z +1 \right) P_n^\prime \left( z \right) \\
&= -\sqrt{2n+1} \cdot t^{-1} \left[ nP_n(z) + (2n-1)P_{n-1}(z) + (2n-3)P_n(z)+ \cdots \right] \\
&= -t^{-1} \sqrt{2n+1} \cdot \left[ \frac{n}{\sqrt{2n+1}}\cdot g_n(t,x) + \sqrt{2n-1}\cdot g_{n-1}(t,x) + \sqrt{2n-3}\cdot g_{n-2}(t,x)+\cdots \right]
\end{aligned}
$$

---

## HiPPO-LegS の導出 (論文Appendix D.3)
### 係数のダイナミクス 1

HiPPO-LegT と同様に求める。
$g_n(t,t) = \sqrt{2n+1} \cdot P_n(1) = \sqrt{2n+1}$ を用いると、

$$
\begin{aligned}
\frac{d}{dt} c_n(t) =& \int f(x) \left( \frac{\partial}{\partial t} g_n (t,x) \right) \omega (t,x) dx \\ &+ \int f(x) g_n (t,x) \left( \frac{\partial}{\partial t} \omega (t,x) \right) \\
=& -t^{-1}\sqrt{2n+1} \cdot \left[ \frac{n}{\sqrt{2n+1}}\cdot c_n(t) + \sqrt{2n-1} \cdot c_{n-1}(t) + \sqrt{2n-3}\cdot c_{n-2}(t) + \cdots \right] \\
&- t^{-1}c_n(t) + t^{-1}f(t)g_n(t,t) \\
=& -t^{-1}\sqrt{2n+1}\cdot \left[ \frac{(n+1)}{\sqrt{2n+1}}\cdot c_n(t) + \sqrt{2n-1}\cdot c_{n-1}(t) + \sqrt{2n-3}\cdot c_{n-2}(t) + \cdots \right] \\
&+ t^{-1}\sqrt{2n+1}\cdot f(t)
\end{aligned}
$$

---

## HiPPO-LegS の導出 (論文Appendix D.3)
### 係数のダイナミクス 2
liner ODE の形にまとめると、
$$
\begin{aligned}
\frac{d}{dt} c(t) &= -\frac{1}{t}A c(t) + \frac{1}{t} Bf(t) \\
A_{nk} &=
\begin{cases}
  \sqrt{2n+1}\cdot \sqrt{2k+1} &\text{if}& n>k \\
  n+1 &\text{if}& n=k \\
  0 &\text{if}& n<k
\end{cases} \\
B_n &= \sqrt{2n+1}
\end{aligned}
$$

---

## HiPPO-LegS の導出 (論文Appendix D.3)
### 係数のダイナミクス 3
liner ODE を行列形式で示す。
$$
\frac{d}{dt}c(t) = -t^{-1}D \left[MD^{-1}c(t)+\bold{1}f(t) \right]
$$
ここで、$D:=\text{diag}\left[\sqrt{2n+1} \right]^{N-1}_{n=0}$ 、$\bold{1}$ は要素がすべて1のベクトル、$M$ は
$$
M =
\begin{bmatrix}
1      & 0      & 0      & 0      & \cdots & 0      \\
1      & 2      & 0      & 0      & \cdots & 0      \\
1      & 3      & 3      & 0      & \cdots & 0      \\
1      & 3      & 5      & 4      & \cdots & 0      \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
1      & 3      & 5      & 7      & \cdots & N
\end{bmatrix},\quad
M_{nk} = \begin{cases}
2k+1 &\text{if}& k<n \\
k+1 &\text{if}& k=n \\
0 &\text{if}& k>n
\end{cases}
$$

---

## HiPPO-LegS の導出 (論文Appendix D.3)
### 得た係数から入力信号 $f$ を近似する関数を再構成する時

任意の時間 $t$ において、
$$
\begin{aligned}
f(x) \approx g^{(t)}(x)&= \sum_n c_n(t)g_n (t,t) \\
&= \sum_n c_n(t) \sqrt{2n+1} \cdot P_n \left( 2\frac{x}{t} - 1 \right)
\end{aligned}
$$

---

## HiPPO-LegS 理論的特徴 (論文Appendix E)
### 時間スケールへのロバスト性
$\tilde{f}(t) = f(\alpha t),\ c = \text{proj}f,\ \tilde{c}=\text{proj}\tilde{f}$ とおく。
ここで、 $\alpha>0$ は時間スケールの引き伸ばし度合い。
$$
\begin{aligned}
\tilde{c}(t) &= \langle \tilde{f},g^{(t)}_n \rangle_{\mu^{(t)}} \\
&= \int \tilde{f}(t) \sqrt{2n+1}\cdot P_n \left( 2\frac{x}{t}-1 \right) \frac{1}{t} \mathbb{I}_{[0,1]} \left( \frac{x}{t}\right) dx \\
&= \int f(\alpha t) \sqrt{2n+1} \cdot P_n \left( 2\frac{x}{t}-1 \right) \frac{1}{t} \mathbb{I}_{[0,1]} \left( \frac{x}{t} \right) dx \\
&= \int f(\alpha t) \sqrt{2n+1}\cdot P_n \left( 2\frac{x}{\alpha t}-1\right) \frac{1}{\alpha t} \mathbb{I}_{[0,1]} \left( \frac{x}{\alpha t} \right) dx \\
&= c_n(\alpha t)
\end{aligned}
$$
3行目→4行目の $=$ は、 $[0,t]$ の範囲から $[0,\alpha t]$ の範囲に変えただけ。