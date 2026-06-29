# CWG-E — Lý thuyết chi tiết (bản đầy đủ)

Bản này viết **đầy đủ**: bài toán, phát biểu định lý chặt, và **chứng minh
từng bước**. Ba kết quả lõi (Prop A, B.1, C) có chứng minh hoàn chỉnh ở
mức nêu rõ; Prop B.2 và Định lý hội tụ hạt (§6) chỉ có chiến lược chi
tiết, **cần chuyên gia OT đồng kiểm** — đánh dấu rõ ở §7.

> Bản này là tham chiếu kỹ thuật cho `docs/proposal.md`. Các định lý
> được rút gọn-phát biểu trong `proposal.md §5` đều trỏ về số hiệu ở
> đây (Định lý A / B / C, Hệ quả A.1, A.2, B.3, …).

---

## 0. Ký hiệu

- Không gian $\mathbb R^d$, chi phí bậc hai $C(x,y)=\tfrac12\|x-y\|^2$.
- Nhân Gibbs (kernel) ở thang điều hoà $\varepsilon>0$:
  $k_\varepsilon(x,y)=e^{-C(x,y)/\varepsilon}=e^{-\|x-y\|^2/(2\varepsilon)}\in(0,1]$.
- $\mathcal P_2(\mathbb R^d)$: độ đo xác suất mômen bậc 2 hữu hạn. Đích $p$, sinh $q$.
- Với độ đo $\mu$ và điểm $x$, **chiếu barycentric Gibbs**:
$$
\tau_\mu(x)\;=\;\frac{\displaystyle\int y\,k_\varepsilon(x,y)\,\mu(\mathrm dy)}{\displaystyle\int k_\varepsilon(x,y)\,\mu(\mathrm dy)}\;=\;\frac{\mathbb E_{Y\sim\mu}[\,k_\varepsilon(x,Y)\,Y\,]}{\mathbb E_{Y\sim\mu}[\,k_\varepsilon(x,Y)\,]}\in\mathbb R^d .
\tag{0.1}
$$
Đây là hồi quy Nadaraya–Watson tự chuẩn hoá (one-step Sinkhorn, marginal
cột tự do). Phân biệt với **Sinkhorn đầy đủ** (thêm thế đối ngẫu cột $g$:
$k_\varepsilon\mapsto k_\varepsilon\,e^{g(y)/\varepsilon}$) — phần lý
thuyết §2 chứng minh cho dạng Gibbs (0.1); mở rộng Sinkhorn đầy đủ ở §7.
- Vận tốc debiased (W-Flow): $V_\varepsilon[q](x)=\tau_p(x)-\tau_q(x)$, là
  $-\nabla_{W_2}$ của phiếm hàm $q\mapsto S_\varepsilon(q,p)$ (Sinkhorn divergence).

**Tính chất nền (trích dẫn, Feydy et al. 2019; W-Flow):** với nhân
$k_\varepsilon$ xác định dương (đúng cho chi phí bậc hai),
$S_\varepsilon(q,p)\ge0$, $=0\iff q=p$, và vận tốc Sinkhorn-divergence
triệt tiêu **chỉ** tại $q=p$. Ta dùng tính chất này như tiên đề nền (P0).

---

## 1. Bài toán

Ta nghiên cứu **cluster-wise barycentric velocity** như một *estimator*
của trường vận tốc, và hỏi: phân cụm đánh đổi gì giữa **phương sai
mini-batch**, **điểm cân bằng**, và **độ chệch**?

**Mô hình hỗn hợp.** Đích là hỗn hợp $K$ thành phần:
$$
p=\sum_{k=1}^K\pi_k\,p_k,\qquad \pi\in\Delta^{K-1},\quad \operatorname{supp}(p_k)=S_k,
\tag{1.1}
$$
với (khi cần) khoảng cách tách $\delta=\min_{k\ne l}\operatorname{dist}(S_k,S_l)$.
Giả định giá bị chặn: $\operatorname{diam}\big(\bigcup_k S_k\big)\le R<\infty$.

**Hai estimator** cho điểm nguồn $x$ với $c(x)=k_0$ (cụm của $x$), dùng
mini-batch $Y_1,\dots,Y_M\stackrel{iid}\sim p$, đặt $w_j:=k_\varepsilon(x,Y_j)$:
$$
\widehat T_{\mathrm{glob}}(x)=\frac{\sum_{j=1}^M w_jY_j}{\sum_{j=1}^M w_j}
\quad\text{(toàn cục)},\qquad
\widehat T_{\mathrm{clus}}(x)=\frac{\sum_{j:\,Y_j\in S_{k_0}} w_jY_j}{\sum_{j:\,Y_j\in S_{k_0}} w_j}
\quad\text{(cluster-wise).}
\tag{1.2}
$$

**Giả thiết A1 (phân tách & khôi phục cụm).** $\delta>0$ và bộ phân cụm
gán đúng thành phần: $Y\in S_k\Rightarrow$ nhãn $=k$, và $c(x)=k_0$ với
$x\in S_{k_0}$.

Các câu hỏi: (Q1) phương sai của hai estimator? (Q2) khi nào trường vận
tốc cluster-wise có điểm cân bằng giả? (Q3) độ chệch giữa cluster-wise
và toàn cục?

---

## 2. Định lý A — Phương sai mini-batch (chứng minh đầy đủ, bậc nhất)

### 2.1 Bổ đề A0 (hiệp phương sai tiệm cận của ước lượng tỉ số)

**Bổ đề A0.** Cho $Y_1,\dots,Y_M\stackrel{iid}\sim\mu$, $w_j=k_\varepsilon(x,Y_j)$,
và $\widehat T_M(x)=\frac1M\sum w_jY_j\big/\frac1M\sum w_j$. Đặt
$m_D=\mathbb E_\mu[w]>0$, $\tau=\tau_\mu(x)$. Khi đó
$$
\sqrt M\big(\widehat T_M(x)-\tau\big)\;\xrightarrow{d}\;\mathcal N\!\big(0,\;\Sigma_\mu(x)\big),\qquad
\Sigma_\mu(x)=\frac{\mathbb E_\mu\!\big[\,w^2\,(Y-\tau)(Y-\tau)^\top\big]}{\big(\mathbb E_\mu[w]\big)^2}.
\tag{2.1}
$$

**Chứng minh.** Viết $\widehat T_M=g(\bar N_M,\bar D_M)$ với
$\bar N_M=\frac1M\sum w_jY_j$, $\bar D_M=\frac1M\sum w_j$, và
$g(N,D)=N/D$. Vì $w\le1$ và $\mathbb E\|Y\|^2<\infty$, các mômen bậc hai
của véc-tơ $(wY,w)\in\mathbb R^{d+1}$ hữu hạn. CLT đa biến:
$$
\sqrt M\begin{pmatrix}\bar N_M-m_N\\\bar D_M-m_D\end{pmatrix}\xrightarrow{d}\mathcal N(0,\Lambda),\quad
\Lambda=\operatorname{Cov}\!\big((wY,\,w)\big)=\begin{pmatrix}\Lambda_{NN}&\Lambda_{ND}\\\Lambda_{ND}^\top&\Lambda_{DD}\end{pmatrix},
$$
với $m_N=\mathbb E[wY]=m_D\tau$. Hàm $g$ khả vi tại $(m_N,m_D)$, Jacobian
$$
J=\Big[\tfrac{\partial g}{\partial N}\;\big|\;\tfrac{\partial g}{\partial D}\Big]=\Big[\tfrac1{m_D}I_d\;\big|\;-\tfrac{m_N}{m_D^2}\Big]=\Big[\tfrac1{m_D}I_d\;\big|\;-\tfrac{\tau}{m_D}\Big].
$$
Phương pháp delta: $\sqrt M(\widehat T_M-\tau)\xrightarrow{d}\mathcal N(0,J\Lambda J^\top)$.
Khai triển:
$$
J\Lambda J^\top=\tfrac1{m_D^2}\big(\Lambda_{NN}-\tau\Lambda_{ND}^\top-\Lambda_{ND}\tau^\top+\tau\tau^\top\Lambda_{DD}\big).
$$
Dùng $m_N=m_D\tau$ và đồng nhất thức then chốt $\mathbb E[w(Y-\tau)]=m_N-m_D\tau=0$,
ta tính trực tiếp (thế $\Lambda_{NN}=\mathbb E[w^2YY^\top]-m_Nm_N^\top$, v.v.,
rồi rút gọn các số hạng $\tau\tau^\top$):
$$
\Lambda_{NN}-\tau\Lambda_{ND}^\top-\Lambda_{ND}\tau^\top+\tau\tau^\top\Lambda_{DD}
=\mathbb E\big[w^2(Y-\tau)(Y-\tau)^\top\big].
$$
*(Kiểm: $\operatorname{Cov}(w(Y-\tau))=\mathbb E[w^2(Y-\tau)(Y-\tau)^\top]$
vì $\mathbb E[w(Y-\tau)]=0$; khai triển vế phải trùng từng số hạng với
vế trái sau khi thay $m_N=m_D\tau$.)* Chia $m_D^2$ ra (2.1). $\square$

### 2.2 Định lý A (phương sai hai estimator)

**Định lý A.** Dưới A1, với $x\in S_{k_0}$:
$$
M\cdot\operatorname{Cov}\big[\widehat T_{\mathrm{glob}}(x)\big]\to\Sigma_{\mathrm{glob}}(x)=\Sigma_p(x),
\qquad
M\cdot\operatorname{Cov}\big[\widehat T_{\mathrm{clus}}(x)\big]\to\frac{1}{\pi_{k_0}}\,\Sigma_{p_{k_0}}(x),
\tag{2.2}
$$
trong đó $\Sigma_\mu(x)$ cho bởi (2.1), $\tau$ tương ứng là $\tau_p(x)$
và $\tau_{p_{k_0}}(x)$.

**Chứng minh.** Vế trái: áp Bổ đề A0 với $\mu=p$. Vế phải: số đích rơi
vào $S_{k_0}$ là $M_{k_0}\sim\operatorname{Binom}(M,\pi_{k_0})$, và **có
điều kiện** trên $\{Y_j\in S_{k_0}\}$ chúng iid theo $p_{k_0}$ (do A1,
nhãn xác định bởi giá). Áp A0 với $\mu=p_{k_0}$ và cỡ mẫu $M_{k_0}$:
$M_{k_0}\operatorname{Cov}\to\Sigma_{p_{k_0}}$. Vì $M_{k_0}/M\xrightarrow{a.s.}\pi_{k_0}$
(LLN), quy về thang $M$ cho thừa số $1/\pi_{k_0}$; dao động nhị thức của
$M_{k_0}$ là bậc thấp hơn $1/M$. $\square$

### 2.3 Hệ quả A.1 (điều kiện thắng — tường minh)

**Hệ quả A.1.** Cluster-wise giảm phương sai (vết) tại $x\in S_{k_0}$
**khi và chỉ khi**
$$
\boxed{\;\operatorname{tr}\Sigma_{p_{k_0}}(x)\;<\;\pi_{k_0}\,\operatorname{tr}\Sigma_{p}(x)\;}
\tag{2.3}
$$

### 2.4 Bổ đề A2 (giới hạn tách-xa: hai phương sai TRÙNG nhau)

**Bổ đề A2.** Dưới A1, khi $\varepsilon/\delta^2\to0$:
$$
\Sigma_p(x)\;=\;\frac1{\pi_{k_0}}\Sigma_{p_{k_0}}(x)\;\big(1+O(e^{-\delta^2/(2\varepsilon)})\big).
$$

**Chứng minh.** Với $y\in S_l,\ l\ne k_0$: $w=k_\varepsilon(x,y)\le e^{-\delta^2/(2\varepsilon)}$.
Tách cả tử $\mathbb E_p[w^2(Y-\tau_p)(\cdot)^\top]$ và mẫu $\mathbb E_p[w]$
theo cụm: phần $l\ne k_0$ có khối lượng $\le e^{-\delta^2/(2\varepsilon)}$
và toạ độ bị chặn bởi $R$, nên $\mathbb E_p[w]=\pi_{k_0}\mathbb E_{p_{k_0}}[w](1+O(e^{-\delta^2/2\varepsilon}))$
và $\tau_p=\tau_{p_{k_0}}+O(e^{-\delta^2/2\varepsilon})$; tương tự tử
$=\pi_{k_0}\,\mathbb E_{p_{k_0}}[w^2(Y-\tau_{p_{k_0}})(\cdot)^\top](1+O(\cdot))$.
Lấy tỉ số: $\Sigma_p=\frac{\pi_{k_0}(\cdots)}{\pi_{k_0}^2(\cdots)}=\frac1{\pi_{k_0}}\Sigma_{p_{k_0}}(1+O(e^{-\delta^2/2\varepsilon}))$. $\square$

### 2.5 Diễn giải (đây là phần "lý thuyết giải thích dữ liệu")

Kết hợp (2.3) và A2 cho bức tranh **đúng** (lưu ý: khác trực giác "càng
tách càng lợi"):

- **Giới hạn $\varepsilon\to0$ (tách xa tuyệt đối): KHÔNG có lợi** — hai
  phương sai trùng nhau (A2). Lợi thế cluster-wise là **hiện tượng
  $\varepsilon$ hữu hạn** (mode chồng lấn ở thang $\varepsilon$).
- **Tử số lợi ích:** ở $\varepsilon$ vừa, nhân $k_\varepsilon$ gán trọng
  số đáng kể cho mode khác ⟹ $\tau_p$ bị kéo *ra giữa* các mode ⟹
  $\|Y-\tau_p\|$ lớn lên cho $Y\in S_{k_0}$ và có khối lượng trên $Y$
  liên-cụm ⟹ $\operatorname{tr}\Sigma_p$ phồng.
- **Mẫu số phạt:** vế trái (2.3) chia hữu hiệu cho $\pi_{k_0}$ — **phạt
  cỡ mẫu**, lớn khi có **nhiều cụm nhỏ**.

Do đó điều kiện (2.3) thắng $\Leftrightarrow$ *(phồng do chồng-mode)* $>$
*(phạt $1/\pi_{k_0}$)*. Khớp dữ liệu:
- **ring8 / ring_minority** ($K=8$, $\pi_{k_0}\approx1/8$, mode tách
  vừa): phạt nhỏ, phồng đủ ⟹ **giảm phương sai** (std W² $0.013$ vs $0.047$). ✓
- **grid25** ($K=25$, $\pi_{k_0}\approx1/25$): phạt $1/\pi_{k_0}\approx25$
  lấn át ⟹ **hoà**. ✓

> Đây là một **định lý điều kiện** kèm ngưỡng tường minh — chính tính
> điều kiện giải thích vì sao thắng ở chỗ này, hoà ở chỗ kia, thay vì
> hứa hão "luôn tốt hơn".

---

## 3. Định lý B — Đặc trưng điểm cân bằng (chứng minh đầy đủ, mức tổng thể)

Dùng vận tốc cluster-wise **trần** (Γ = I) với **cân bằng marginal trong
cụm**: trong mỗi cụm $k$, OT chạy giữa các *phân phối chuẩn hoá*
$\hat q_k:=q|_{S_k}/m^q_k$ và $\hat p_k:=p|_{S_k}/m^p_k$, với
$m^q_k=q(S_k),\ m^p_k=p(S_k)$. Vận tốc trong cụm $k$:
$$
V_k=\tau_{\hat p_k}-\tau_{\hat q_k}\quad(\text{trên }S_k),
$$
là vận tốc Sinkhorn-divergence giữa $\hat q_k$ và $\hat p_k$.

**Định lý B.1 (cân bằng giả của cluster-trần).**
$V\equiv0$ (trên toàn $\operatorname{supp} q$) **khi và chỉ khi**
$\hat q_k=\hat p_k$ cho mọi $k$. Đặc biệt, nếu các *hình dạng trong cụm*
trùng ($\hat q_k=\hat p_k\ \forall k$) nhưng *khối lượng cụm* lệch
($m^q\ne m^p$), thì $V\equiv0$ trong khi $q\ne p$ — một **điểm cân bằng giả**.

**Chứng minh.** Cân bằng marginal cưỡng bức tách bài toán thành $K$ bài
OT độc lập trên các cặp chuẩn hoá $(\hat q_k,\hat p_k)$; không số hạng
nào phụ thuộc $m^q_k,m^p_k$ (đã chia ra). Trên $S_k$, $V_k$ là vận tốc
Sinkhorn-divergence; theo (P0), $V_k\equiv0\iff\hat q_k=\hat p_k$. Vậy
$V\equiv0\iff\hat q_k=\hat p_k\ \forall k$. Vì $q=\sum_k m^q_k\hat q_k$
và $p=\sum_k m^p_k\hat p_k$, đẳng thức hình dạng $\hat q_k=\hat p_k\ \forall k$
**không** kéo theo $q=p$ trừ khi cũng có $m^q=m^p$. Chọn $q$ với
$\hat q_k=\hat p_k$ nhưng $m^q\ne m^p$ ⟹ $q\ne p$, $V\equiv0$. $\square$

**Hệ quả B.3 (cơ chế collapse ở Gate 2).** Đầu huấn luyện, $q$ chưa phủ
mode ⟹ $m^q$ lệch xa $m^p$ (vài cụm $m^q_k\approx0$). Theo B.1, thành
phần *lệch khối lượng* không sinh lực ⟹ không có gradient kéo $q$ phủ
cụm trống ⟹ **tự củng cố collapse**. Khớp `gate2` (hard sụp) và
`thm_no_spurious` (trường cluster-trần $\approx0$ ở mọi $\alpha$).

**Định lý B.2 (outer-Γ khôi phục — phát biểu; chứng minh phác, cần kiểm).**
Thêm số hạng cấp-cụm dùng $\Gamma=\pi^{\varepsilon_\gamma}(m^q,m^p)$
(Sinkhorn trên tâm cụm), vận tốc tổng $V^{+\Gamma}\equiv0\iff q=p$.
*Phác:* $V^{+\Gamma}=0$ buộc (i) $\hat q_k=\hat p_k\ \forall k$ (như B.1)
và (ii) số hạng outer $=0$, mà số hạng này là vận tốc Sinkhorn-divergence
$S_{\varepsilon_\gamma}(m^q,m^p)$ trên đơn hình ⟹ $m^q=m^p$; (i)+(ii)$\Rightarrow q=p$.
*Cần kiểm:* tương tác hai số hạng không tạo nghiệm-zero phụ; điều kiện
trên $\varepsilon_\gamma$. (Xem §7.)

---

## 4. Định lý C — Tính nhất quán (chứng minh đầy đủ)

**Định lý C.** Dưới A1, với $x\in S_{k_0}$ và chi phí bậc hai:
$$
\big\|\tau_p(x)-\tau_{p_{k_0}}(x)\big\|\;\le\;\frac{2R\,e^{-\delta^2/(2\varepsilon)}\,(1-\pi_{k_0})}{\pi_{k_0}\,\kappa_{k_0}(x)}\;=\;O\!\big(e^{-\delta^2/(2\varepsilon)}\big),
\tag{4.1}
$$
với $\kappa_{k_0}(x)=\mathbb E_{p_{k_0}}[k_\varepsilon(x,Y)]$, $R=\operatorname{diam}(\bigcup_kS_k)$.

**Chứng minh.** Viết $A=\int_{S_{k_0}}\!yk_\varepsilon\,p$, $B=\int_{S_{k_0}}\!k_\varepsilon\,p$,
$A'=\int_{\cup_{l\ne k_0}S_l}\!yk_\varepsilon\,p$, $B'=\int_{\cup_{l\ne k_0}S_l}\!k_\varepsilon\,p$.
Thì $\tau_p=\frac{A+A'}{B+B'}$, $\tau_{p_{k_0}}=\frac{A}{B}$. Với $y\notin S_{k_0}$:
$\|x-y\|\ge\delta\Rightarrow k_\varepsilon(x,y)\le e^{-\delta^2/2\varepsilon}$.
Do đó $B'\le e^{-\delta^2/2\varepsilon}(1-\pi_{k_0})$ và $\|A'\|\le R\,B'$
(vì $\|y\|\le R$ sau dời gốc về $x$, dùng $\|y-x\|\le R$ và xét $\tau$
tương đối). Ước lượng hiệu:
$$
\Big\|\tfrac{A+A'}{B+B'}-\tfrac AB\Big\|
=\Big\|\tfrac{B A'-B' A}{B(B+B')}\Big\|
\le\frac{B\|A'\|+B'\|A\|}{B(B+B')}
\le\frac{B'(R+\|\tau_{p_{k_0}}\|)}{B}.
$$
Với gốc toạ độ đặt tại $x$ (tịnh tiến không đổi vận tốc),
$\|A\|/B=\|\tau_{p_{k_0}}-x\|\le R$, nên tử $\le 2R\,B'$. Mà
$B=\pi_{k_0}\kappa_{k_0}(x)$. Thay $B'$ cho (4.1). $\square$

**Diễn giải (bias–variance qua $\varepsilon$).** $\varepsilon$ nhỏ ⟹ độ
chệch (4.1) mũ-nhỏ (nhất quán) **nhưng** phương sai cao (Định lý A: $k$
nhọn). $\varepsilon$ vừa ⟹ chệch lớn hơn, phương sai thấp. Cluster-wise
giữ nhánh chệch ở chế độ tách-xa **và** cải thiện nhánh phương sai (A.1)
⟹ **đẩy đường cong bias–variance**. Đây là luận điểm hợp nhất.

---

## 5. Ví dụ minh hoạ tính được (kiểm chứng Định lý A)

**Ví dụ (hai mode 1-D).** $p=\tfrac12\mathcal N(-a,\sigma^2)+\tfrac12\mathcal N(+a,\sigma^2)$,
$x=-a$ (tâm mode trái), $k_0=$ trái. Với $k_\varepsilon$ Gauss, mọi tích
phân $\mathbb E[w^m Y^n]$ là Gauss đóng-dạng (tích các hàm mũ bậc hai).
Tính $\Sigma_p$ và $\tfrac1{1/2}\Sigma_{p_{k_0}}=2\Sigma_{p_{\text{trái}}}$
rồi so theo (2.3): với $\sigma$ nhỏ và $a/\sqrt\varepsilon$ trung bình,
$\Sigma_p>2\Sigma_{p_{\text{trái}}}$ (cluster-wise thắng); khi
$a/\sqrt\varepsilon\to\infty$ hai vế bằng nhau (A2). *(Tính tay được;
nên đưa vào phụ lục như kiểm chứng giải tích của Định lý A và A2.)*

---

## 6. (Chiến lược, CHƯA chứng minh) Hội tụ động lực hạt → WGF

**Mục tiêu.** Dưới A1 + chính quy,
$x_i^{(t+1)}=x_i^{(t)}+\eta\,V^{+\Gamma}_\varepsilon[q^{(t)}_N](x_i^{(t)})$
hội tụ, khi $\eta\to0,\ N,M\to\infty$, về nghiệm yếu của
$\partial_tq_t+\nabla\!\cdot(q_tV_\varepsilon[q_t])=0$, đều trên $[0,T]$ theo $W_2$.

**Chiến lược (3 mảnh, mỗi mảnh là một mảng giải tích):** (i) ổn định
Lipschitz của chiếu barycentric entropic theo nhiễu marginal (định lý ổn
định Sinkhorn / hội tụ thế đối ngẫu); (ii) cluster-wise = tổng trực tiếp
trên phân hoạch giá + outer-Γ là perturbation hạng-thấp; (iii) ghép qua
lập luận mean-field/compactness cho particle gradient flow (propagation
of chaos).

**Vì sao không viết proof ở đây:** mỗi mảnh dễ có lỗ hổng tinh vi (đặc
biệt (i),(iii)). Khuyến nghị: hoặc (a) đồng tác giả nền OT, hoặc (b)
**rút gọn về định lý W-Flow**: nếu W-Flow đã có định lý hội tụ cho
trường toàn cục, chỉ cần chứng minh phần *điều chỉnh block + outer-Γ*
là perturbation bị chặn — phạm vi hẹp, khả thi hơn.

---

## 7. Bảng minh bạch độ chặt

| Kết quả | Độ chặt | Phụ thuộc / việc thêm |
|---|---|---|
| Bổ đề A0, Định lý A, Hệ quả A.1, Bổ đề A2 | **Proof đầy đủ** (tiệm cận bậc nhất, delta-method chuẩn) | Bản hữu-hạn-mẫu (concentration của ước lượng tỉ số); chuyển Gibbs (0.1) → **Sinkhorn đầy đủ** (kiểm soát dao động đối ngẫu cột $g$) |
| Định lý B.1 + Hệ quả B.3 | **Proof đầy đủ** (mức tổng thể) | Dựa (P0) — tính divergence của $S_\varepsilon$ (Feydy/W-Flow); bản empirical |
| Định lý B.2 (outer-Γ) | **Phác** | Kiểm tương tác hai số hạng; điều kiện $\varepsilon_\gamma$ |
| Định lý C | **Proof đầy đủ** (Gibbs, tách-xa) | Bản Sinkhorn đầy đủ; hằng số tường minh hơn |
| Ví dụ §5 | Tính tay được | Hoàn tất tích phân Gauss, đưa vào phụ lục |
| §6 (hội tụ hạt → WGF) | **Chỉ chiến lược** | Cần chuyên gia OT, hoặc rút gọn về W-Flow |

**Phụ thuộc nền (phải trích dẫn rõ):** (P0) tính positive-definite của
nhân $e^{-C/\varepsilon}$ và tính divergence của Sinkhorn $S_\varepsilon$
(Feydy et al. 2019); định lý hội tụ trường toàn cục của W-Flow (cho §6
rút gọn).

---

## 8. Dàn ý paper lý thuyết-trọng tâm

1. Mở đầu: cluster-wise entropic OT như *velocity estimator*; câu hỏi đánh đổi.
2. Thiết lập (§0–§1).
3. **Định lý A + A.1 + A2** (kết quả chính): phương sai, ngưỡng thắng
   tường minh, giới hạn tách-xa. Kèm Ví dụ §5.
4. **Định lý B** (đặc trưng cân bằng): cân bằng giả của cluster-trần
   (B.1), cơ chế collapse (B.3), vai trò outer-Γ (B.2).
5. **Định lý C** (nhất quán) + bias–variance qua $\varepsilon$.
6. Thực nghiệm *xác nhận lý thuyết* (không cần thắng SOTA): multi-seed
   variance ↔ A.1/A2; `thm_no_spurious` ↔ B; `thm_consistency` ↔ C;
   gate2 collapse ↔ B.3. Trình bày kết quả âm về generation **như dự
   đoán của lý thuyết**.
7. Bàn luận & giới hạn (Gibbs vs Sinkhorn; §6 mở).
