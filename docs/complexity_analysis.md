# Complexity Analysis of JSA Algorithm with MIS

本文分析 **Joint Stochastic Approximation (JSA)** 训练过程的计算复杂度，其中重点是
**Metropolis Independent Sampling (MIS)** 在前向与反向阶段所带来的计算复杂度。

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

> 我们这里假设，带梯度和不带梯度的前向时间相同。
> 至于显存占用方面，带梯度的往往会比不带梯度的高，因为需要存储中间激活以便反向传播。一般而言，显存会高出 2-3 倍左右，但这取决于具体模型结构与实现细节。

假设：

* $q_\phi(h\mid x)$ 为 **categorical 分布**
* $p_\theta(x\mid h)$ 为 **Gaussian decoder**
* $T_q \approx T_p$（encoder / decoder 规模对称）

---

## 2. 标准 JSA + MIS 算法流程

### 2.1 算法步骤

1. MIS 采样 $h \sim p_\theta(h\mid x)$：

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
2. 计算梯度 $\nabla_\theta \log p_\theta(x,h)$ 和 $\nabla_\phi \log q_\phi(h\mid x)$
3. 参数更新 $\theta^{(new)} \leftarrow \theta^{(old)} + \eta \nabla_\theta$, $\phi^{(new)} \leftarrow \phi^{(old)} + \eta \nabla_\phi$

---

## 3. 原始 JSA 的时间复杂度

### 3.1 MIS 单步复杂度

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

### 3.2 MIS 总复杂度

$$
O\big(T_{\text{init}} + L(2T_p + 3T_q)\big)
$$

若 $T_p \approx T_q$，则：

$$
O(5LT_q)
$$

> 即：一次 MIS 等价于 **几十次 encoder / decoder 前向**，

### 3.3 JSA 算法总复杂度

JSA 训练中，我们上面的 MIS 采样过程中，都是不带梯度的前向计算。后续更新参数，需要对采样出来的 $h$，重新带梯度进行前向计算，便于反向传播。这里又有两种选择：

1. 只使用 $L$ 步 MIS 采样出来的最终 $h_L$，进行一次带梯度前向
2. 使用 MIS 过程中所有的 $h_1, \dots, h_L$，进行 $L$ 次带梯度前向

假设我们采用第一种方式，则使用 $h_L$ 进行带梯度前向的复杂度为 $O(T_p + T_q)$。（计算一次 $\log p_\theta(x,h_L)$ 和 $\log q_\phi(h_L\mid x)$）。因此整体复杂度为：
$$
O\big(T_{\text{init}} + L(2T_p + 3T_q) + (T_p + T_q)\big)
$$

如果使用第二种方式，我们还需要对所有的 $h_1, \dots, h_L$ 进行带梯度前向，复杂度为 $O(L(T_p + T_q))$。整体复杂度为：
$$
O\big(T_{\text{init}} + L(3T_p + 4T_q)\big)
$$

> 在一次迭代中，我们假设 $L=10$，则光 MIS 采样，需要进行约 $50$ 次 encoder / decoder 前向计算，而后面的参数迭代，如果使用第二种方式，又需要额外进行 $20$ 次前向计算。整体计算代价非常高。


---

## 4. 第一步优化：MIS 采样中共享 $q_\phi(h\mid x)$

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

> 我曾经将其展开为:
> $$
> \begin{align*}
\log r &= -\frac{1}{2\sigma^2}\left[\left\Vert {x_i-g_\theta(h_i)}\right\Vert _2^2-\left\Vert {x_i-g_\theta(h_i^{(t-1)})}\right\Vert _2^2\right]-\left[f_\phi(h_i)-f_\phi(h_i^{(t-1)})\right]\\
&= \frac{1}{\sigma^2}\left(g_\theta(h_i) -  g_\theta(h_i^{(t-1)})\right)^T \left( x_i - \frac{g_\theta(h_i) + g_\theta(h_i^{(t-1)})}{2} \right) - \left(f_\phi(x_i)[h_i]-f_\phi(x_i)[h_i^{(t-1)}]\right) 
\end{align*}
> $$
> 发现这样在数学上是完全没问题的，但是工程上，数值不稳定，容易导致训练崩溃。
> 我们在训练过程中，得到的 $\mu$ 和 $\mu'$ 往往非常接近 $x$，导致作差后的结果非常小，进而在浮点数运算中丢失精度。

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

## 9. 工程实现中的问题

### 9.1 复杂度总结

通过 **共享 encoder、缓存 decoder、index 化状态**
可将 MIS 复杂度从
$$
O\big(L(2T_p + 3T_q)\big)
$$
降至
$$
O\big(2T_q\big)
$$

而后续参数更新，我们如果使用所有的 $h_1, \dots, h_L$，计算 $\log q_\phi(h\mid x)$ 时，我们可以并行计算，因为只需要一次 encoder 前向，不同的 $h$，只是使用的 logit 不同，复杂度为 $O(T_q)$。计算 $\log p_\theta(x,h)$ 时，我们也可以 batch 化 decoder 前向，复杂度为 $O(T_p)$。但是此时出现问题，如果我们 batch 化 decoder 前向，我们本身也会有一个 batch size 的限制，则一次性计算，相当于把 batch size 乘以 $L$，实际操作过程中，会发生显存爆炸。因此，在实际代码中，我会将采样过程分块（chunk）进行，计算完 loss 后直接梯度反传，以控制显存使用峰值。此时复杂度与采样步数 $L$ 呈正比。极限情况，当 chunk size 设为 1 时，复杂度退化回 $O\big(LT_q\big)$。

总结起来，我们目前的 JSA + MIS 实现复杂度为：
$$
O\big(T_q + T_p\big) + O\big(T_q + \frac{L}{C}T_p\big)
$$
其中 $C$ 为 chunk size。

### 9.2 MIS 采样带梯度？

我们上面的做法是，MIS 采样阶段不带梯度，仅在后续参数更新阶段，使用采样结果带梯度前向计算。是否可以直接在 MIS 采样阶段，带梯度前向？理论上是可以的，但是实践中会遇到一些问题：

* 显存占用过高：MIS 采样过程中，每一步都需要存储中间激活以便反向传播，显存使用会成倍增长，难以控制
* 数值不稳定：MIS 采样过程中，接受率计算涉及多个前向结果的作差，带梯度时，容易导致数值不稳定，训练崩溃

因此，目前我们仍然采用不带梯度的 MIS 采样，后续再进行带梯度前向计算的方式。

