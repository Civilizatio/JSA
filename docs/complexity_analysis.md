# Complexity Analysis

这里主要分析一下，JSA 在训练过程中，进行前向以及反向的复杂度。特别是 Metropolis Independent Sampling (MIS) 的复杂度。

## 算法流程

假设我们设置的超参数为：采样步数为 L。这里只针对一个样本。
我们假设，proposal model 一次前向需要的时间为 $T_q$，计算联合概率 $p_{\theta}(x,h)$ 需要的时间为 $T_p$。

那么，采样 $h\sim q_{\phi}(h|x)$ 的时间为 $T_q$。（先进行前向，得到 logits，之后的采样步骤与前向相比几乎不耗时间）

下面是标准化的算法流程：

1. initialize $h_0$:

    - if use_cache, sample from cache;
    - else sample from $q_{\phi}(h|x)$

2. for t = 0, 1, ..., L-1:

    1. sample $h'$ from $q_{\phi}(h|x)$
    2. compute acceptance ratio $a = \frac{p_{\theta}(x,h')q_{\phi}(h_t|x)}{p_{\theta}(x,h_t)q_{\phi}(h'|x)}$
    3. sample u from Uniform(0,1)
    4. if u < a, set $h_{t+1} = h'$, else set $h_{t+1} = h_t$

## 复杂度分析

1. 初始化 $h_0$:

    - if use_cache, O(1)
    - else O($T_q$)

2. for t = 0, 1, ..., L-1:

    - sample $h'$ from $q_{\phi}(h|x)$: O($T_q$)
    - compute acceptance ratio $a = \frac{p_{\theta}(x,h')q_{\phi}(h_t|x)}{p_{\theta}(x,h_t)q_{\phi}(h'|x)}$: O($2T_p + 2T_q$)
    - sample u from Uniform(0,1): O(1)
    - if u < a, set $h_{t+1} = h'$, else set $h_{t+1} = h_t$: O(1)

Overall complexity: 

$$
O(T_{init} + L * (2T_p + 3T_q))
$$

也就是说，举个简单的例子，我们设置 L=10，那么每次进行 MIS 采样的时间大概是 $O(10 * (2T_p + 3T_q))$。如果 $T_p$ 和 $T_q$ 差不多的话（encoder和decoder大概是对称的，假设往往成立），那么大概是 $O(50T_q)$。也就是说，大概相当于前向 50 次的时间，显然会导致时间复杂度很高

## 如何简化

我们假设，encoder网络为 $f_{\phi}(x)$，decoder网络为 $g_{\theta}(h)$。并且假设 $p_{\theta}(x|h)$ 是高斯分布。因此对数接受概率为：
$$
\begin{align*}
\log r &= -\frac{1}{2\sigma^2}\left[\left\Vert {x_i-g_\theta(h_i)}\right\Vert _2^2-\left\Vert {x_i-g_\theta(h_i^{(t-1)})}\right\Vert _2^2\right]-\left[f_\phi(h_i)-f_\phi(h_i^{(t-1)})\right]\\
&= \frac{1}{\sigma^2}\left(g_\theta(h_i) -  g_\theta(h_i^{(t-1)})\right)^T \left( x_i - \frac{g_\theta(h_i) + g_\theta(h_i^{(t-1)})}{2} \right) - \left(f_\phi(x_i)[h_i]-f_\phi(x_i)[h_i^{(t-1)}]\right)
\end{align*}
$$

我们的拒绝概率为 $a = \min(1, \exp(\log r))$。


### 共享 $q_{\phi}(h|x)$ 的计算

关于 $q_{\phi}(h|x)$ 的计算方式，模型输入 $x$，得到 logits。采样的时候，直接利用 logits 进行 `softmax`，采样。如果计算相关的概率，也是利用这里的 logits 进行计算。

而这里的所有的 $x$，都是一个tensor，这意味着，我们其实只需要计算一次 $q_{\phi}(h|x)$。在计算接受概率的时候，将这个 logits 送入，直接 logits 作差取指数就是所需的概率比值。

同时，其实这里的每一步迭代的采样，并不需要上一步的 $h$，因此是可以并行采样的，也就是说，一次性采样 L 个 $h'$，然后运行循环，每次从这 L 个 $h'$ 中取出一个进行计算即可。

算法变为：（为了方便，这里假设 use_cache=False）
1. compute $q_{\phi}(h|x)$, get logits
2. sample $h_0, h_1', ..., h_{L-1}', h_{L}'$ from $q_{\phi}(h|x)$
3. for t = 0, 1, ..., L-1:

    1. compute acceptance ratio $a = \frac{p_{\theta}(x,h_{t+1}')q_{\phi}(h_t|x)}{p_{\theta}(x,h_t)q_{\phi}(h_{t+1}'|x)}$
    2. sample u from Uniform(0,1)
    3. if u < a, set $h_{t+1} = h_{t+1}'$, else set $h_{t+1} = h_t$

此时的复杂度变为：
Overall complexity:

$$
O(T_{init} + T_q + L * (2T_p))
$$


### $p_{\theta}(x,h)$ 的近似计算

关于 $p_{\theta}(x,h)$ 的计算方式，模型输入 $h$，得到重构的 $x'$，由于这里是高斯假设，那么计算 $p_{\theta}(x|h)$ 的概率，实际上是计算 $||x - x'||^2$。这个计算量比较大。

目前的做法是，在计算 acceptance ratio 的时候，只需要算：

$$
\frac{1}{\sigma^2}\left(g_\theta(h_i) -  g_\theta(h_i^{(t-1)})\right)^T \left( x_i - \frac{g_\theta(h_i) + g_\theta(h_i^{(t-1)})}{2} \right)
$$

其中 $g_\theta(h)$ 是 decoder 的前向输出。
每次都要计算两次前向，计算量比较大。

但其实，我们发现，同样的，这里的 $x$ 始终是一个固定的值。因此，只要 $h$ 的值不变，我们的 $g_\theta(h)$ 其实是不用算的。但是，我们有一个最基本的认识，就是我们至少要把所有的 $g_\theta(h)$ 都算一遍，才能进行比较。

因此，我们可以把所有的 $h$ 都先算一遍 $g_\theta(h)$，存起来。然后在计算 acceptance ratio 的时候，直接从存储中读取 $g_\theta(h)$ 的值，进行计算。

算法变为：（为了方便，这里假设 use_cache=False）

1. compute $q_{\phi}(h|x)$, get logits
2. sample $h_0, h_1', ..., h_{L-1}', h_{L}'$ from $q_{\phi}(h|x)$
3. compute $g_\theta(h_0), g_\theta(h_1'), ..., g_\theta(h_{L}')$
4. for t = 0, 1, ..., L-1:
    1. compute acceptance ratio $a = \frac{p_{\theta}(x,h_{t+1}')q_{\phi}(h_t|x)}{p_{\theta}(x,h_t)q_{\phi}(h_{t+1}'|x)}$
    2. sample u from Uniform(0,1)
    3. if u < a, set $h_{t+1} = h_{t+1}'$, else set $h_{t+1} = h_t$

此时的复杂度变为：
Overall complexity:
$$
O(T_{init} + T_q + (L+1) * T_p)
$$
也就是说，现在每次进行 MIS 采样的时间大概是 $O((L+1) * T_p + T_q)$。如果 $T_p$ 和 $T_q$ 差不多的话（encoder和decoder大概是对称的，假设往往成立），那么大概是 $O((L+2) * T_q)$。也就是说，大概相当于前向 $(L+2)$ 次的时间，显然相比之前的 $O(L * (2T_p + 3T_q))$ 有了很大的提升。

## 伪代码

```python
def mis_sampling(x, L):
    # x: input data, shape: (batch_size, data_dim)
    # L: number of MIS steps

    # Step 1: compute q_phi(h|x)
    logits = encoder(x) # shape: (batch_size, num_latent_vars, num_categories)

    # Step 2: sample h0, h1', ..., hL' from q_phi(h|x)
    h_samples = sample_from_logits(logits, L + 1) # shape: (batch_size, L + 1, num_latent_vars)

    # Step 3: compute g_theta(h) for all sampled h
    g_outputs = decoder(h_samples.view(-1, num_latent_vars)) # shape: (batch_size * (L + 1), data_dim)
    g_outputs = g_outputs.view(batch_size, L + 1, data_dim)

    # Step 4: MIS iterations
    h_current = h_samples[:, 0, :] # initial h0

    for t in range(L):
        h_proposal = h_samples[:, t + 1, :]

        # compute acceptance ratio
        log_r = compute_log_acceptance_ratio(x, h_current, h_proposal, logits, g_outputs)
        a = torch.exp(log_r).clamp(max=1.0)

        # sample u from Uniform(0, 1)
        u = torch.rand(batch_size)

        # accept or reject
        h_current = torch.where(u < a, h_proposal, h_current)

    return h_current
```

其中，`compute_log_acceptance_ratio` 函数计算接受概率的对数形式，利用预先计算好的 encoder logits 和 decoder outputs。

这里的一个问题就是，我再计算 acceptance ratio 的时候，如何像 encoder logits 一样，只需要 `torch.gather` 就能得到所需的概率值。对于 decoder outputs，我们可以直接利用 `h_current` 和 `h_proposal` 的索引，直接从 `g_outputs` 中取出对应的重构结果，进行计算。




