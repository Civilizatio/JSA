# JSA with Perceptual Loss

主要介绍在 vanilla JSA 的基础上如何引入 perceptual loss 的。JSA 需要我们显式建模联合似然 $\log p(x,h)$，用于计算损失，以及计算接受概率。

> 损失方面公式如下：
> $$ \nabla_\theta \log p_\theta(x) = \mathbb{E}_{p_\theta (h|x)}\left[ \nabla_\theta \log p_\theta (x, h) \right] $$
> 接受概率方面如下：
> $$ r=\frac{p_\theta(h_i,x_i)/q_\phi(h_i|x_i)}{p_\theta(h_i^{(t-1)},x_i)/q_\phi(h_i^{(t-1)}|x_i)} $$

我们为了方便，显式地使用了一个 decoder 来建模条件似然 $\log p(x\mid h)$，并假设其为高斯分布，而 $\log p(h)$ 认为是均匀分布。
因此最终的损失，只有 MSE 这样的形式。
然而，高斯的假设带来的问题是，模型更倾向于学习像素级别的平均信息，对于一些边缘、棱角这样我们感知上差距很大的反而建模很少。为了解决这个问题，在 VQ-GAN 中，在重建损失中引入了 perceptual loss，并使用 L1 范数。

$$
\mathcal L_{\text{rec}}=
\|x - \hat x\|_1  
+  
\lambda_{\text{perc}}  
\|\phi(x) - \phi(\hat x)\|_2
$$
 VQ-GAN 中的 perceptual loss，特指使用 LPIPS 网络，得到 VGG 网络的中间层特征 $\phi$，用于约束。但其实后面的语音的重建损失，还引入了类似 Mel 谱的 L1 范数，L2 范数，以及特征网络的中间层特征，以及将这些特征加和。为了统一建模，我们统一将这些提取原始数据 $x$ 的高级特征的操作定义为 $\phi$，则加入 perceptual loss 后的完整的形式应该是：

$$
\mathcal L_{\text{rec}}=
\sum_p\lambda_p\|x - \hat x\|_p
+  
\sum_i \sum_p\beta_{i,p}  
\|\phi_i(x) - \phi_i(\hat x)\|_p
$$
> 上面是最一般的形式，就是包括不同的 $p$ 范数，以及不同的特征提取网络（需要注意，这里的提取特征的网络都是 stop gradient 的，不参与训练）。$\lambda_p$，$\beta_{i,p}$ 为权重。为了方便，我们记前面的损失统一为 pixel loss:
> $$ \text{PixelLoss}(x, \hat{x})=\sum_p\lambda_p\|x - \hat x\|_p $$
> 后面的 perceptual loss 为：
> $$ \text{PerceptualLoss}(x, \hat{x})=\sum_i \sum_p\beta_{i,p} \|\phi_i(x) - \phi_i(\hat x)\|_p $$

上面是 VQ-GAN 的建模方式，其定义的重建损失是没有显式的概率定义的。但是我们的 JSA 是需要显式定义联合似然的，这也给我们带来了困难。

我们之前定义的 Gaussian 分布作为似然，是一种很取巧的行为，因为其本身已经满足归一化了。但是如果我们直接迁移进来，定义分布为：
$$
\log p_\theta(x\mid h) = -\text{PixelLoss}(x, \hat{x})
-\text{PerceptualLoss}(x, \hat{x})+\log Z(h)
$$
会发现，这里前面的并不是归一化的，我们不得不考虑一个归一化常数的问题。更困难的是，我们在计算接受概率时，一旦有了归一化常数，将会出现 $\log \left(Z(h_{i+1})-Z(h_i)\right)$ 这样无法计算的问题。

## 1. 联合能量模型定义

上面说，我们定义条件似然会遇到归一化常数难以估计的问题。因此我们可以直接定义全局似然：
$$
\log p_\Theta(x, h) = -\text{PixelLoss}(x, \hat{x}_\theta)
-\text{PerceptualLoss}(x, \hat{x}_\theta)-f_\psi(h)+\text{Constant}
$$

其中 $\Theta=\{\theta,\psi \}$
相当于定义了一个能量模型：
$$
p_\Theta(x,h)=\frac{1}{Z(\Theta)}\exp\left(-u_\Theta(x,h)\right)
$$

为了方便，我们定义：
$$
d_\theta(x,h) = \text{PixelLoss}(x, \hat{x}_\theta)
+\text{PerceptualLoss}(x, \hat{x}_\theta)
$$
表示失真度的一种度量。
> 注意，这里的 $\hat{x}=g_\theta (h)$，我们还可以用一个decoder来实现，但是其已经不是一个重建的意义了，而是能量的一项，只不过我们通过能量的定义，提出了 $x$ 应该逼近 $g_\theta (h)$ 的假设。

则：
$$
u_\Theta(x,h) = d_\theta (x,h) + f_\psi(h)
$$
> 我们在定义了一个 $g_\theta$ 用于计算失真度量外，还需要一个计算 $f(h)$ 的能量的网络，由于是离散序列，很自然的，我们可以定义一个 transformer，计算这个序列的能量：
> $$ f_\psi(\mathbf{h}_{1:L})=-\log p_\psi (\mathbf{h}_{1:L}) = -\sum_{i=1}^L \log p_\psi(h_i\mid h_{<i},\text{BOS}) $$
> 需要注意的是，这里的 $f_\psi$ 的建模不应该忽略或者视为均匀分布，因为一旦为均匀分布，就变为联合分布满足能量为 $d_\theta$ 了，这显然是很奇怪的。

## 2. 优化目标

我们希望最大化边缘似然 $\log p_{\Theta} (x)$，
即为：
$$
\max_\Theta \mathbb{E}_{\tilde{p}(x)}\left[\log p_\Theta(x)\right]
$$
则可以定义损失为：
$$
\mathcal{L}_\Theta = - \mathbb{E}_{\tilde{p}(x)}\left[\log p_\Theta(x)\right]
$$
写出其梯度：
$$
\nabla_{\theta,\psi}\mathcal{L}_\Theta  = \mathbb{E}_{\tilde{p}(x)p_{\theta,\psi}(h\mid x)}\left[\nabla_{\theta,\psi} u_{\theta,\psi} (x,h)\right]-\mathbb{E}_{p_{\theta,\psi}(x,h)}\left[\nabla_{\theta,\psi} u_{\theta,\psi} (x,h)\right]
$$
分别写出 $\theta,\psi$ 的梯度：
$$
\nabla_{\theta} \mathcal{L}_\Theta = \mathbb{E}_{\tilde{p}(x)p_{\theta,\psi}(h\mid x)}\left[\nabla_{\theta} d_{\theta} (x,h)\right]-\mathbb{E}_{p_{\theta,\psi}(x,h)}\left[\nabla_{\theta} d_{\theta} (x,h)\right]
$$
$$
\nabla_{\psi} \mathcal{L}_\Theta = \mathbb{E}_{\tilde{p}(x)p_{\theta,\psi}(h\mid x)}\left[\nabla_{\psi} f_{\psi} (h)\right]-\mathbb{E}_{p_{\theta,\psi}(x,h)}\left[\nabla_{\psi} f_{\psi} (h)\right]
$$
问题的关键在于，我们需要获得来自 $p_{\theta,\psi}(x,h)$ 的样本。

## 3. 采样算法

我们需要采样的样本，$x$ 为连续数据，$h$ 为离散的序列。关于连续的数据，可以用 LD 采样。我们利用下面的公式：
$$
x_{\tau+1} = x_\tau + \frac{\epsilon^2}{2}\nabla_x\log p(x_\tau) +\epsilon \xi_\tau
$$
$\xi_\tau\sim\mathcal{N}(0,I)$，$\epsilon$ 为步长。

在这里，就是：
$$
x_{\tau+1} = x_\tau - \frac{\epsilon^2}{2}\nabla_x d_\theta(x,h)+\epsilon \xi_\tau
$$
也就是说，我们对 $x$ 的采样只需要关注 $d_\theta$ 这一项，因为 $\nabla_x f_\psi (h)=0$。

对于 $h$ 的采样，由于 $h$ 为离散的序列，目前可以有下面的两种思路：

1. 延续之前的 JSA 的做法，我们引入一个辅助的分布 $q_\phi (h|x)$，对于序列的建模也是假设条件独立，然后使用 MIS (Metropolis Independent Sampling)。
    我们来讨论这种方案的可行性：

    我们的目标分布为：
    $$
    \begin{align*}
    p_\Theta (h\mid x)&\propto \exp(-u_\Theta(x,h))\\
    &=\exp(-d_\theta(x,h)-f_\psi(h))
    \end{align*}
    $$
    则更新 $\phi$ 的损失为：
    $$
    \mathcal{L}_\phi = -\mathbb{E}_{\tilde{p}(x)p_{\theta,\psi}(h\mid x)}\left[\log q_\phi(h\mid x)\right]
    $$
    $$
    \begin{align*}
    \nabla_\phi \mathcal{L}_\phi &=-\mathbb{E}_{\tilde{p}(x)p_{\theta,\psi}(h\mid x)}\left[\nabla_\phi \log q_\phi(h\mid x)\right]
    \end{align*}
    $$
    提议分布为 $q_\phi (h\mid x)$，计算接受概率：
    $$
    \begin{align*}
    \log r &= \log \frac{p_\theta(h_i,x_i)}{p_\theta(h_i^{(t-1)},x_i)}-\log\frac{q_\phi(h_i|x_i)}{q_\phi(h_i^{(t-1)}|x_i)}\\
    &=-\left[d_\theta(x_i,h_i)-d_\theta(x_i,h_i^{(t-1)})+f_\psi(h_i)-f_\psi(h_i^{(t-1)})\right] - \left[\log q_\phi(h_i\mid x_i)- \log q_\phi(h_i^{(t-1)}\mid x_i)\right]
    \end{align*}
    $$
    其中 $\log q_\phi$ 为辅助分布的 logits.

    为了简写，我们定义下面的变化：
    $$
    \begin{align}  \Delta d_{\theta,i,t} &\triangleq d_\theta(x_i,h_i) - d_\theta(x_i,h_i^{(t-1)}) \\  
    \Delta f_{\psi,i,t} &\triangleq f_\psi(h_i) - f_\psi(h_i^{(t-1)}) \\  
    \Delta \log q_{\phi,i,t} &\triangleq \log q_\phi(h_i\mid x_i) - \log q_\phi(h_i^{(t-1)}\mid x_i) \end{align}
    $$
    则
    $$
    \log r(h_i^{(t-1)}, h_i) = -\Delta d_{\theta,i,t}-\Delta f_{\psi,i,t}-\Delta \log q_{\phi,i,t}
    $$
    这三项都是可以计算的。

2. 采用 NCG （Norm Constrained Gradient sampler）的思路，同样是每个位置 $h$ 的提议，利用梯度提议。

    来讨论这种方法的可行性。
    提议分布可以设置为：
    $$
    q(h'\mid h)\propto \exp \left(-\frac{1}{2}\nabla u_\Theta(h)(h'-h)-\frac{1}{2\alpha}\Vert h'-h\Vert_p^p\right)
    $$
    这里需要对 $h$ 求导，而能量包括：$d_\theta(x,h)+f_\psi(h)$ 两部分。
    > 这里需要说明，由于 $h$ 是离散的，我们求导的对象应该是 $h$ 对应的 embedding，即 $\nabla_{e(h)}$，而非 $\nabla_{h}$。后面我们默认就是对 embedding 求导。

    关于 $f_\psi$ 的部分，我们的 transformer 已经定义了概率，能量也是能直接求导得到的。而前者 $d_\theta(x,h)$ 对 $h$ 求导 ，其实就是一个卷积网络的求导，也是可以得到的。每个位置的更新，其实也可以单独进行，即写为：
    $$
    q(h'\mid h)\propto\prod_{n=1}^{N} \exp\left(-\frac{1}{2} \nabla_n u_\Theta(h)^\top (h'_n - h_n) - \frac{1}{2\alpha} \|h'_n - h_n\|_p^p\right)
    $$

    然后分别求导，更新即可。
    具体的困难以及解决方案，在 [NCG in JSA perceptual loss](./ncg_in_jsa_perceptual.md) 中有详细的介绍。

    一个问题就是 NCG 需要一个好的初始状态来提议，如果从噪声开始提议，则收敛需要比较慢的时间。或许这两种可以结合使用？先用 MIS 提议一个还可以的分布，然后用梯度信息做进一步的更正？

## 4. 最终算法

输入：数据 $\tilde{x}$，模型参数 $\Theta=\{\theta,\psi\}$，辅助分布参数 $\phi$，采样步数 $T$，MIS proposal 数量 $K$，NCG采样步长 $\alpha$，范数参数 $p$，NCG 采样步数 $L$，LD 步长 $\epsilon$。

循环直到收敛：

1. **Positive phase:** $(\tilde{x},h)\sim \tilde{p}(x)p_{\theta,\psi}(h\mid x)$

    1. 初始化 $h^{(0)}$，从 cache 中或者 $q_\phi(h\mid \tilde{x})$ 采样得到，并计算 $w^{(0)}$。
    2. 对于 $t=1$ 到 $K$：

        1. MIS proposal 生成 $1$ 个候选：
           $$h' \sim q_\phi(h\mid \tilde{x})$$
        2. 计算 importance weight：
           $$w^{(t)} = \frac{p_\Theta(\tilde{x},h')}{q_\phi(h'\mid \tilde{x})}$$
        3. 计算接受率：
           $$r = \min\left(1, \frac{w^{(t)}}{w^{(t-1)}}\right)=\min\left(1, \frac{p_\Theta(\tilde{x},h')q_\phi(h^{(t-1)}\mid \tilde{x})}{p_\Theta(\tilde{x},h^{(t-1)})q_\phi(h'\mid \tilde{x})}\right)$$
        4. 以概率 $r$ 接受 $h'$，否则保持 $h^{(t)}=h^{(t-1)}$。
        5. (可选) 使用 NCG 对 $h^{(t)}$ 进行进一步的梯度更新，提议分布为：
           $$q(h'\mid h^{(t)})\propto \exp \left(-\frac{1}{2}\nabla u_\Theta(h^{(t)})(h'-h^{(t)})-\frac{1}{2\alpha}\Vert h'-h^{(t)}\Vert_p^p\right)$$
           接受率同样计算。进行 $L$ 步的 NCG 更新。
  
    3. 返回样本 $(\tilde{x},h^{(K)})$。

2. **Negative phase:** $(x,h)\sim p_{\theta,\psi}(x,h)$

    1. 初始化 $(x^{(0)},h^{(0)})$，可以用真实的样本 $\tilde{x}$ 以及 $h\sim q_\phi (h\mid \tilde{x})$ 来初始化（此时为 short run MCMC），也可以用 replay buffer 来储存前面的样本（此时为 persistent MCMC）。
    2. 对于 $t=1$ 到 $T$：

        1. LD 更新 $x\sim p_\Theta(x\mid h)$：
           $$x^{(t)} = x^{(t-1)} - \frac{\epsilon^2}{2}\nabla_x d_\theta(x^{(t-1)},h^{(t-1)})+\epsilon \xi^{(t-1)}$$
        2. (Optional) 接受或拒绝 $x^{(t)}$。
        3. MIS 更新 $h\sim p_\Theta(h\mid x)$：同上面的步骤，得到 $h^{(t)}$。
        4. (Optional) 使用 NCG 对 $h^{(t)}$ 进行进一步的梯度更新，提议分布为：
           $$q(h'\mid h^{(t)})\propto \exp \left(-\frac{1}{2}\nabla u_\Theta(h^{(t)})(h'-h^{(t)})-\frac{1}{2\alpha}\Vert h'-h^{(t)}\Vert_p^p\right)$$
           接受率同样计算，进行 $L$ 步的 NCG 更新。

    3. 返回样本 $(x^{(T)},h^{(T)})$。

3. 利用样本 $(\tilde{x},h^{(K)})$ 和 $(x^{(T)},h^{(T)})$ 更新 $\Theta$，$\phi$。

## 5. 风险评估与瓶颈

1. 计算代价极高：LD，NCG 这些采样算法都需要在 $x\in \mathbb{R}^{C\times H\times W}$ 这样的高维空间中进行梯度计算。 MIS 也会累加开销（目前的实验已经说明，如果是串行的 MIS 会比较慢）。会带来较高的前向/反向的开销
2. 采样的有效性：图像空间的 Langevin 采样可能会收敛慢，同时，如果我们为了节省显存占用，初始化使用类似 Contrastive Divergence 的方式，则会有偏。离散的 $h$ 也会接受率较低。
3. 调参敏感：$\epsilon$ （LD步长），K/T（MCMC采样的步数），$\lambda_p,\beta_{i,p}$（loss 的权重）。以及如果引入 replay buffer，buffer的大小等
4. 规模上的限制：batch size 很难加大。

## 6. 可能的改进方法

我们需要学习的参数分为三个部分：

- $g_\theta$：预测得到 $\hat{x}$，用于 $d_\theta(x,h)$ 失真度度量的计算。
- $p_\psi$：学习先验的 $h$ 的分布 $\log p_\psi (h)$。
- $q_\phi$：估计逼近真实的后验分布 $p_\Theta(h\mid x)$。
如果我们 learning from scratch，这三个部分相互耦合，完全从随机状态开始学习，很有可能不会收敛。

因此一个比较稳妥的方法就是，分阶段训练，分两阶段或者三个阶段。

### 6.1. Stage 1: Decoder Pretraining

这一个阶段的目的是学习一个合理的表示空间，为后面的能量学习提供基础。

忽略 $f_\psi$ ，能量变为：
$$
u_\Theta(x,h) = d_\theta (x,h)
$$
同时，我们优化能量函数的时候，忽略 negative phase。

优化目标为：
$$
\nabla_{\theta} \mathcal{L}_\Theta = \mathbb{E}_{\tilde{p}(x)p_{\theta}(h\mid x)}\left[\nabla_{\theta} d_{\theta} (x,h)\right]
$$
$$
\begin{align*}
\nabla_\phi \mathcal{L}_\phi &=-\mathbb{E}_{\tilde{p}(x)p_{\theta}(h\mid x)}\left[\nabla_\phi \log q_\phi(h\mid x)\right]
\end{align*}
$$

其中，后验分布是一个有偏的分布：
$$
\tilde p_\theta(h\mid x) \propto \exp (-d_\theta (x,h))
$$
> 由于我们忽略了归一化常数 $\log Z_\theta (h)$，得到的后验分布也不是上面的目标

训练流程：
输入：数据 $\tilde{x}$，模型参数 $\theta$，辅助分布参数 $\phi$，MIS proposal 数量 $K$。
循环直到收敛：

1. 采样得到 $(\tilde{x},h)$，其中 $h$ 的采样使用 MIS 进行采样，具体步骤如下：

    1. 初始化 $h^{(0)}$，从 cache 中或者 $q_\phi(h\mid \tilde{x})$ 采样得到，并计算 $w^{(0)}$。
    2. 对于 $t=1$ 到 $K$：

        1. MIS proposal 生成 $1$ 个候选：
            $$h' \sim q_\phi(h\mid \tilde{x})$$
        2. 计算 importance weight：
            $$w^{(t)} = \frac{\exp(-d_\theta(\tilde{x},h'))}{q_\phi(h'\mid \tilde{x})}$$
        3. 计算接受率：
            $$r = \min\left(1, \frac{w^{(t)}}{w^{(t-1)}}\right)=\min\left(1, \frac{\exp(-d_\theta(\tilde{x},h'))q_\phi(h^{(t-1)}\mid \tilde{x})}{\exp(-d_\theta(\tilde{x},h^{(t-1)}))q_\phi(h'\mid \tilde{x})}\right)$$
        4. 以概率 $r$ 接受 $h'$，否则保持 $h^{(t)}=h^{(t-1)}$。

    3. 返回样本 $(\tilde{x},h^{(K)})$。
2. 利用样本 $(\tilde{x},h^{(K)})$ 更新 $\theta$，$\phi$。

#### 6.1.1 问题以及改进

碰到的问题是，失真度量 $d_\theta (x,h)$ 的尺度，我们目前就是使用 L2 范数以及 Perceptual feature 的欧式距离。但是有个问题，就是光使用这个范数，存在约束不够的问题，也就是第一阶段的训练，会存在后验分布 $p_\theta (h\mid x)$ 太过平坦，然后同一个 $x$ 会对应多种可能的 $h$，这些 $h$ 的概率都差不多，而且很低。此时第二阶段的学习也会受到影响，同一张图片可能对应多种可能的 $h$，则 transformer 学习的序列也是不稳定的。

我们的目的就是，让第一阶段的模型学习到尽可能稳定、尖锐的分布，然后第二阶段也能学到较好的先验，最后第三阶段完整的 EBM 训练。
为此，我手动调整了权重，即变为了 $\alpha d_\theta(x,h)$，这个系数我设置为 $5$，结果确实比原来的好。我现在在思考，能不能将这个权重设置为可学习的温度，让模型自己收敛到一个比较合适的解，即定义：
$$
u_\Theta(x,h) = \frac{d_\theta (x,h)}{T} + f_\psi (h)
$$
其中 $T$ 是一个可学习的正数，我们可以通过其调整第一项失真度的尺度，来决定后验分布 $p(h\mid x)$ 的尖锐程度。

需要考虑的问题是，放到第三阶段，可以直接通过正负项来解决训练，$T$ 不会变得很大。但是第一阶段我们原来是忽略负项的，那么如果我们直接优化
$$
u_\Theta(x,h) = \frac{d_\theta (x,h)}{T}
$$
损失设置为：
$$
\nabla_{\theta} \mathcal{L}_\Theta = \mathbb{E}_{\tilde{p}(x)p_{\theta}(h\mid x)}\left[\nabla_{\theta,T} \frac{d_{\theta} (x,h)}{T}\right]
$$
此时会发现，模型会趋向于 $T\to\infty$。我们需要找到一个负项，或者说正则项。
应该如何找？

因此，需要回答的问题有：
1. 我第一阶段预训练得到的后验分布以及 proposal model，需不需要是 sharp 的分布？还是说 soft 的分布也是可以的。

2. 能不能将 $T$ 设置为可学习的？如果引入 $T$，第一阶段的损失函数如何设置？

目前的一个策略是，引入一个控制温度的 $T$，在其他地方需要计算能量的时候，使用 $d_\theta (x,h)/T$，在优化损失的时候，引入正则项：
$$
\nabla_{\theta} \mathcal{L}_\Theta = \mathbb{E}_{\tilde{p}(x)p_{\theta}(h\mid x)}\left[\nabla_{\theta,T} \left(\frac{d_{\theta} (x,h)}{T} + c \log T\right)\right]
$$
> 这里 $\log T$ 只 起到一个正则项的作用，放置训练使得 $T\to\infty$ ，而不需要我们在能量中引入定义，因为我们第三阶段的 EBM 训练会通过引入负项自动校准。
> 这里的 $c$ 设置为一个权重，可以为 $C\times H \times W$。（这是为了模仿 Gaussian 分布的方差的权重）

修改后的第一阶段的策略为：

引入一个参数，`temperature`，或者叫做 `distortion_temperature` ，以及引入一个调控的参数，为 `temperature_mode`，可选择的有 `fixed` or `learnable` or `scheduled`，（和 JSA 的对齐，目前先不设置为 `scheduled`，即如果参数为 scheduled 先报错）。为了降低修改的难度，目前的 `distortion_weight` 先留着，后面可以就固定为一个常数。对于引入的温度，和 JSA 的高斯的方差一样，在 joint model 里面保存一个 `log_distortion_temperature`，来保证其一定是正数。
其他的计算能量的部分逻辑不变，还是直接用  `distortion/temperature`，但是在计算损失的时候，即调用 `get_loss`，里面需要引入正则化项 `c log temperature`，来防止温度趋于无穷大。请你按照我上面的要求修改代码。

### 6.2. Stage 2: Prior Pretraining

 固定好第一阶段的模型，即固定好 $q_\phi (h\mid x)$ 和 $d_\theta (x,h)$，只用第二阶段的 transformer 学习一个隐空间的 $h$ 的分布。尽管这里的 $h$ 的分布是有偏的，但是我们只需要一个稳定的 $f_\psi (h)$ 给第三阶段。

则修改后的优化目标为：
$$\begin{align*}
\mathcal{L}_\psi &= -\mathbb{E}_{\tilde{p}(x)\tilde p(h\mid x)}\left[\log p_\psi(h)\right] \\
&= -\mathbb{E}_{\tilde{p}(x)\tilde p(h\mid x)}\left[\sum_{i=1}^L \log p_\psi(h_i\mid h_{<i},\text{BOS})\right]
\end{align*}
$$
其中 $\tilde p(h\mid x) \propto \exp (-d_\theta (x,h))$ 是第一阶段训练得到的一个有偏的后验分布。则此时只需要标准的 LLM 自回归的交叉熵损失就可以计算。

训练流程：

输入：数据 $\tilde{x}$，模型参数 $\psi$，第一阶段训练得到的 $q_\phi$ 和 $d_\theta$，MIS proposal 数量 $K$。
循环直到收敛：

1. 编码得到 $(\tilde{x},h)$，其中 $h$ 的采样使用 $q_\phi (h\mid x)$ 得到。
2. 利用样本 $(\tilde{x},h)$ 更新 $\psi$。

### 6.3. Stage 3: Full EBM Training

完整的 EBM，由于前两个阶段已经学习了较好的编码器、解码器已经先验估计器，这里可以引入负项。

优化目标和训练流程同上面的[最终算法](#4-最终算法)。

### 6.4. NCG 的使用

NCG 理论上在第一阶段以及第三阶段都可以使用，第二阶段由于我们固定好了 $q_\phi$ 和 $d_\theta$，直接编码、解码得到 $h$ 就可以了，不需要采样。（也可以设置为需要采样，但是只需要 MIS 采样即可）。
第一阶段我们只需要计算 $d_\theta$ 的梯度即可，第三阶段需要计算 $u_\Theta$ 的梯度。
