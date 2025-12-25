# Complexity Analysis of MIS in JSA

本文分析 **Joint Stochastic Approximation (JSA)** 训练过程中，
**Metropolis Independent Sampling (MIS)** 在前向与反向阶段所带来的计算复杂度，并给出一套在**工程上可落地的复杂度优化方案**。

重点关注以下问题：

* MIS 在原始形式下为何计算代价极高
* 哪些计算可以共享、缓存或 batch 化
* 如何在不改变算法正确性的前提下，将复杂度降到可接受范围

---

## 1. 问题设置与符号约定

考虑单个样本 $x$，隐变量为离散向量 $h$。

* proposal 分布：$q_\phi(h\mid x)$
* target 分布：$p_\theta(x,h) = p_\theta(x\mid h)p(h)$

设：

* MIS 迭代步数：$L$
* encoder（proposal）一次前向时间：$T_q$
* decoder（或 joint likelihood）一次前向时间：$T_p$

假设：

* $q_\phi(h\mid x)$ 为 **categorical 分布**
* $p_\theta(x\mid h)$ 为 **Gaussian decoder**
* $T_q \approx T_p$（encoder / decoder 规模对称）

---

## 2. 标准 MIS 算法流程（单样本）

### 2.1 算法步骤

1. 初始化 $h_0$

   * 若使用 cache：$O(1)$
   * 否则从 $q_\phi(h\mid x)$ 采样

2. 对 $t = 0, \dots, L-1$：

   1. 采样 $h' \sim q_\phi(h\mid x)$
   2. 计算接受率
      $$
      r
      =
      \frac{
      p_\theta(x,h')q_\phi(h_t\mid x)
      }{
      p_\theta(x,h_t)q_\phi(h'\mid x)
      }
      $$
   3. 以 $\min(1,r)$ 接受或拒绝

---

## 3. 原始 MIS 的时间复杂度

### 3.1 单步复杂度

在一次 MIS 迭代中，需要：

| 操作                   | 复杂度   |
| -------------------- | ----- |
| 采样 $h'\sim q_\phi$   | $T_q$ |
| 计算 $p_\theta(x,h')$  | $T_p$ |
| 计算 $p_\theta(x,h_t)$ | $T_p$ |
| 计算 $q_\phi(h')$      | $T_q$ |
| 计算 $q_\phi(h_t)$     | $T_q$ |

总计：

$$
O(2T_p + 3T_q)
$$

### 3.2 总复杂度

$$
O\big(T_{\text{init}} + L(2T_p + 3T_q)\big)
$$

若 $T_p \approx T_q$，则：

$$
O(5LT_q)
$$

> 即：一次 MIS 等价于 **几十次 encoder / decoder 前向**，
> 在 JSA 训练中是不可接受的。

---

## 4. 第一步优化：共享 $q_\phi(h\mid x)$

### 4.1 关键观察

* 对同一个 $x$，所有 proposal 来自 **同一个分布**
* encoder logits **只需计算一次**
* 所有 $\log q_\phi(h\mid x)$ 都可以通过 `gather` 得到

### 4.2 优化后算法

1. encoder 前向一次，得到 logits
2. 一次性采样：
   $$
   h_0, h_1', \dots, h_L' \sim q_\phi(h\mid x)
   $$
3. MIS 过程中仅做：

   * decoder 前向
   * logits 索引
   * 常数时间计算

### 4.3 复杂度

$$
O\big(T_{\text{init}} + T_q + L(2T_p)\big)
$$

---

## 5. 第二步优化：缓存 decoder 输出

### 5.1 Gaussian decoder 的结构

对于 Gaussian 假设：

$$
\log p_\theta(x\mid h)
=
-\frac{1}{2\sigma^2}
\left\lVert x - g_\theta(h) \right\rVert^2

+ C
  $$

关键事实：

* 若 $h$ 不变，$g_\theta(h)$ 不需要重复计算
* MIS chain 只会在有限集合
  ${h_0, h_1', \dots, h_L'}$ 中跳转

---

### 5.2 优化后的完整流程

1. encoder 前向一次：$T_q$
2. 一次性采样 $L+1$ 个 latent
3. **batch 化 decoder 前向**：
   $$
   {g_\theta(h_0), g_\theta(h_1'), \dots, g_\theta(h_L')}
   $$
4. MIS 阶段只进行：

   * index 操作
   * 向量运算

---

### 5.3 最终复杂度

$$
O\big(T_{\text{init}} + T_q + T_p\big)
$$

若 $T_p \approx T_q$：

$$
O\big(2T_q\big)
$$

> 这是在 **不改变 MIS 正确性** 的前提下，
> JSA + MIS 能达到的 **基本最优复杂度**。

---

## 6. 关于“并行性”的澄清

* proposal 采样：**可并行**
* encoder / decoder 前向：**可并行**
* accept / reject 决策：**必须串行**

因此：

> 并行的是「候选生成」，
> 串行的是「Markov chain 的状态更新」。

---

## 7. 接受率的实现形式（推荐）

采用最稳定、最直接的形式：

$$
\log r
=

-\frac{1}{2\sigma^2}
\left(
\left\lVert x - \mu' \right\rVert^2 - \left\lVert x - \mu \right\rVert^2
\right)
+
\log q(h_t\mid x)
-

\log q(h'\mid x)
$$

其中：

* $\mu = g_\theta(h_t)$
* $\mu' = g_\theta(h')$

避免不必要的代数展开，有利于数值稳定。

---

## 8. 可直接实现的 MIS 伪代码

### 8.1 核心设计原则

> **MIS 的状态不是 $h$，而是「候选集合中的 index」**

这样可以：

* logits：`torch.gather`
* decoder 输出：`index_select`
* 无需比较 tensor 或使用 hash 表

---

### 8.2 伪代码

```python
def mis_sampling(x, L):
    """
    x: (B, data_dim)
    returns: h_final (B, num_latent_vars)
    """

    B = x.size(0)

    # 1. Encoder forward (shared)
    logits = encoder(x)                     # (B, V, K)
    log_q = torch.log_softmax(logits, dim=-1)

    # 2. Sample candidates
    h_all = sample_from_logits(logits, L + 1)  # (B, L+1, V)

    # 3. Decoder forward (batched)
    h_flat = h_all.view(B * (L + 1), -1)
    recon_flat = decoder(h_flat)
    recon = recon_flat.view(B, L + 1, -1)

    # 4. log q(h|x)
    log_q_all = torch.gather(
        log_q.unsqueeze(1).expand(-1, L + 1, -1, -1),
        dim=-1,
        index=h_all.unsqueeze(-1)
    ).sum(dim=(-1, -2))                     # (B, L+1)

    # 5. log p(x|h)
    sq_err = ((x.unsqueeze(1) - recon) ** 2).sum(dim=-1)
    log_px_all = -0.5 * sq_err / sigma2     # (B, L+1)

    # 6. MIS chain over indices
    cur_idx = torch.zeros(B, dtype=torch.long, device=x.device)

    for t in range(L):
        prop_idx = torch.full_like(cur_idx, t + 1)

        log_r = (
            log_px_all.gather(1, prop_idx[:, None]).squeeze(1)
            - log_px_all.gather(1, cur_idx[:, None]).squeeze(1)
            + log_q_all.gather(1, cur_idx[:, None]).squeeze(1)
            - log_q_all.gather(1, prop_idx[:, None]).squeeze(1)
        )

        accept = torch.rand(B, device=x.device) < torch.exp(log_r).clamp(max=1.0)
        cur_idx = torch.where(accept, prop_idx, cur_idx)

    # 7. Return final h
    h_final = h_all.gather(
        1, cur_idx[:, None, None].expand(-1, 1, h_all.size(-1))
    ).squeeze(1)

    return h_final
```

## 9. 总结

* 原始 MIS 在 JSA 中计算代价极高
* 通过 **共享 encoder、缓存 decoder、index 化状态**
  可将复杂度从
  $$
  O\big(L(2T_p + 3T_q)\big)
  $$
  降至
  $$
  O\big(2T_q\big)
  $$
* 核心思想：

  > **把 Markov chain 的状态从「变量值」变成「候选索引」**

也因此，我们增加采样步数 $L$ 时，训练时间几乎不变，这使得 JSA + MIS 在实际应用中变得可行。因为对于 index 的串行拒绝接受，计算代价极低。
