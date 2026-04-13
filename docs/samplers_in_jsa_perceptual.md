# Samplers in JSA + Perceptual Loss

主要介绍 NCG 采样器以及 LD 采样器在当前的能量设置下如何使用的问题。

## NCG Sampler

> 这里内容完全来自 [rde tokenizer](./rde_tokenizer.md#232-p-ncg)

> 由于这里需要区分离散的序列和序列中的元素，我们用 $\mathbf{h}$ 来表示一个 token 序列，$\mathbf{h} = (h_1, h_2, ..., h_L)$，其中 $L$ 是序列长度，每个位置上的 token 取值范围为 $[1, K]$。

我们需要从如下的分布中采样：
$$\pi_\theta(\mathbf{h}) \propto e^{-u_\theta(\mathbf{h})}$$

对于连续的能量模型，我们通常使用 Langevin 动力学来进行采样，而对于离散的 $\mathbf{h}$，我们没有办法直接对离散变量进行求导，因此我们转化为对 $\mathbf{h}$ 对应的 embedding 进行提议更新。定义 $e_{\mathbf{h}} = (e_{h_1}, e_{h_2}, ..., e_{h_L})$ 为 $\mathbf{h}$ 的位置级 embedding 表示，则我们可以在 embedding 空间计算梯度，利用梯度进行更新。

我们定义的提议规则为：对于当前的 $\mathbf{h}$，提议分布定义为：
$$q(e_{\mathbf{h}'}\mid e_{\mathbf{h}})\propto \exp \left(-\frac{1}{2}\nabla u_\theta(\mathbf{h})^\top(e_{\mathbf{h}'}-e_{\mathbf{h}})-\frac{1}{2\alpha}\Vert e_{\mathbf{h}'}-e_{\mathbf{h}}\Vert_p^p\right)$$
其中，$\alpha$ 为更新的步长，$p$ 为约束的 $p$ 范数，为两个超参数。

上面的提议分布是未归一化的离散分布，其归一化常数难以计算（需要遍历所有长度为 L 的序列，计算复杂度为 $O(K^L)$），我们可以逐位置更新，分解为：
$$q(e_{\mathbf{h}'}\mid e_{\mathbf{h}}) = \prod_{i=1}^L q(e_{h'_i}\mid e_{\mathbf{h}})$$
其中每个位置的提议分布为：
$$q(e_{h'_i}\mid e_{\mathbf{h}})\propto \exp \left(-\frac{1}{2}\nabla_{e_{h_i}} u_\theta(\mathbf{h})^\top(e_{h'_i}-e_{h_i})-\frac{1}{2\alpha}\Vert e_{h'_i}-e_{h_i}\Vert_p^p\right)$$

也就是说，对于每个位置 $i$，我们先计算能量函数关于 $e_{h_i}$ 的梯度 $\nabla_{e_{h_i}} u_\theta(\mathbf{h})$ ，然后遍历整个词表来得到该位置的提议分布：
$$
q(e_{h'_i}\mid e_{\mathbf{h}})= \frac{\exp \left(-s(e_{h'_i}\mid e_{\mathbf{h}})\right)}{\sum_{j=1}^K \exp \left(-s(e_{h'_j}\mid e_{\mathbf{h}})\right)}
$$
其中 $s(e_{h'_i}\mid e_{\mathbf{h}}) = \frac{1}{2}\nabla_{e_{h_i}} u_\theta(\mathbf{h})^\top(e_{h'_i}-e_{h_i})+\frac{1}{2\alpha}\Vert e_{h'_i}-e_{h_i}\Vert_p^p$ 为提议的分数。然后对于 L 个位置，可以按照一定顺序进行 sweep，也可随机给定一个顺序进行提议。

> 需要注意的是，理论上我们的提议都应该是基于 $\mathbf{h}$ 的，但是我们这里直接用 $e_{\mathbf{h}}$ 代替了，认为他们两个是一致的。


当然，为了保证采样的正确性，我们可以加入一个 Metropolis-Hastings 步骤来接受或拒绝这个提议，以确保采样的正确性。接受概率记为 $\rho$：
$$\rho = \min\left(1, \frac{\pi_\theta(e_\mathbf{h}')}{\pi_\theta(e_\mathbf{h})} \cdot \frac{q(e_{\mathbf{h}}\mid e_{\mathbf{h}'})}{q(e_{\mathbf{h}'}\mid e_{\mathbf{h}})}\right)$$
其中 $\mathbf{h}$ 是当前的 token 序列，$\mathbf{h}'$ 是提议得到的 token 序列。通过这种方式，我们可以有效地从能量模型 $p_\theta(\mathbf{h}|x)$ 中采样得到离散 token 序列 $\mathbf{h}$。

需要注意的是，NCG 本质上还是一种 MCMC 方法，我们还需要考虑采样的初始化问题，为此，我们可以采用 MIS ，从一个简单的 proposal distribution 中采样得到初始的 token 序列 $\mathbf{h}^{(0)}$，然后再使用 p-NCG 进行迭代更新。

完整的算法步骤如下：

输入：输入数据 $x$，能量函数 $u_\theta(\mathbf{h})$，提议分布参数 $\alpha$，$p$ 范数参数，辅助 proposal distribution $q_\phi(\mathbf{h}|x)$，迭代次数 $T$。

1. 初始化 $\mathbf{h}^{(0)}$，从 $q_\phi(\mathbf{h}|x)$ 中采样得到。
2. 对于 $t=1$ 到 $T$：

    1. 对于每个位置 $i=1$ 到 $L$ （也可以使用随机顺序进行一遍扫描）：

        1. 计算当前的 embedding $e_{h_i}$。
        2. 根据提议规则计算每个 token 的提议概率：
            $$q(e_{h'_i}\mid e_{\mathbf{h}})\propto \exp \left(-\frac{1}{2}\nabla_{e_{h_i}} u_\theta(\mathbf{h})^\top(e_{h'_i}-e_{h_i})-\frac{1}{2\alpha}\Vert e_{h'_i}-e_{h_i}\Vert_p^p\right)$$
        3. 从提议分布中采样得到新的 token $h'_i$。
    2. 得到新的 token 序列 $\mathbf{h}' = (h'_1, h'_2, ..., h'_L)$。
    3. 计算接受概率 $\rho$：
        $$\rho = \min\left(1, \frac{\pi_\theta(e_\mathbf{h}')}{\pi_\theta(e_\mathbf{h})} \cdot \frac{q(e_{\mathbf{h}}\mid e_{\mathbf{h}'})}{q(e_{\mathbf{h}'}\mid e_{\mathbf{h}})}\right)$$
    4. 接受或拒绝样本：
        - 以概率 $\rho$ 接受 $\mathbf{h}'$，即 $\mathbf{h}^{(t)} = \mathbf{h}'$。
        - 否则拒绝 $\mathbf{h}'$，保持当前样本不变，即 $\mathbf{h}^{(t)} = \mathbf{h}^{(t-1)}$。
3. 返回最终的样本 $\mathbf{h}^{(T)}$。
> NOTE: 上面的我们拒绝接受是在每一次完整的提议之后进行的，但是这样有可能导致接受概率非常低，实际操作过程中，我们可以进行 block 的提议与接受拒绝，即，引入一个参数 `block_size`，每次不必完整的扫过整个 $L$，就作为提议，进行接受拒绝，可以提高接受率。

### NCG in JSA+perceptual loss

上面是一般的 NCG 采样器的做法，具体要如何用到我们的 JSA + perceptual loss 方法中，还需要克服几个难点。

在 JSA + perceptual loss 中，我们的目标分布为：
$$
p_\Theta(\mathbf{h}\mid x)\propto\exp \left(-d_\theta(x,\mathbf{h})-f_\psi(\mathbf{h})\right)
$$
其中，$x\in \mathbb{R}^D$ 是输入的原始数据，$\mathbf{h}$ 是编码后的一个固定长度的离散 token 序列。
能量函数的两项中，第一项：
$$
d_\theta(x,h) = \text{PixelLoss}(x, \hat{x}_\theta)
+\text{PerceptualLoss}(x, \hat{x}_\theta)
$$
其中 $\hat{x}_\theta = g_\theta(\mathbf{h})$ 是通过一个卷积解码器 $g_\theta$ 从 token 序列 $\mathbf{h}$ 解码得到的重建图像。$\text{PixelLoss}$ 是像素级的损失函数，例如 L2 loss，$\text{PerceptualLoss}$ 是感知损失函数，例如  LPIPS loss。（LPIPS 利用一个预训练好的固定权重的 vgg 网络，对原始图像提取不同层的特征，并做加权和，其参数并不会训练）

$f_\psi(\mathbf{h})$ 是一个学习先验的模块，实际中我们利用一个 GPT 模型，输入 token 序列 $\mathbf{h}$，输出一个概率值：

$$ f_\psi(\mathbf{h}_{1:L})=-\log p_\psi (\mathbf{h}_{1:L}) = -\sum_{i=1}^L \log p_\psi(h_i\mid h_{<i},\text{BOS}) $$

那么，NCG 要对这个分布分布采样，主要的难点在于，我们需要计算能量函数 $u_\Theta(\mathbf{h}) = d_\theta(x, \mathbf{h}) + f_\psi(\mathbf{h})$ 对于 $\mathbf{h}$ 的梯度 $\nabla_{\mathbf{h}} u_\theta(\mathbf{h})$，以便在 embedding 空间中进行提议更新。

还记得我们上面在介绍 NCG 的提议时，说我们理论上提议都是基于 $\mathbf{h}$ 的，但是我们直接对其 embedding 进行提议了，认为他们两个是等价的。但其实我们是隐含一个假设，就是这个系统中每个 $h$ 对应一个确定的 $e_h$ 。但是在我们 JSA+perceptual loss中，对同一个 $h$ ，不同的参数部分会有不同的 embedding。

1. 在 $d_\theta(x, \mathbf{h})$ 中，我们是先经过一个 embedding 层，将 token 序列 $\mathbf{h}$ 转换为一个连续的 embedding 表示 $e_{\mathbf{h}}^d$，然后通过一个卷积解码器 $g_\theta$ 将这个 embedding 解码为重建图像 $\hat{x}_\theta$，最后计算像素损失和感知损失。
2. 在 $f_\psi(\mathbf{h})$ 中，我们同样经过一个 embedding 层将 $\mathbf{h}$ 转换为 $e_{\mathbf{h}}^{f,\text{in}}$，然后输入到 GPT 模型中计算语言模型损失。而且在 $f_\psi(\mathbf{h})$ 中，由于我们使用的是自回归模型，我们计算逐位置的 $h_i$ 的梯度，会发现还存在对输出层 embedding $e_{\mathbf{h}}^{f,\text{out}}$ 的梯度。即梯度存在两项：$$\nabla_{e_{h_i}} f_\psi(\mathbf{h}) = - \nabla_{e_{h_i}^\text{out}} \log p_\psi(h_i\mid h_{<i},\text{BOS}) - \sum_{j>i} \nabla_{e_{h_i}^{\text{in}}} \log p_\psi(h_j\mid h_{<j},\text{BOS})$$

两部分的 $h$ 对应不同的 embedding。此时我们不能直接将这两部分的梯度加起来，作为整体的梯度，因为这是两个完全不同的几何空间，维度都可能不同。因此我们不能直接将两个部分的梯度加起来作为最终的梯度。下面是两种实现的思路：

#### 思路1：统一的 embedding

实现的一种思路是，我们强制定义统一的 embedding $e_{h}$，让两个部分共享底层的 embedding，然后分别用一个线性矩阵投影到对应的空间，即：
$$
e_h^d = W^d e_h,\quad e_h^{f,\text{in}} = W^{f,\text{in}} e_h, \quad e_h^{f,\text{out}} = W^{f,\text{out}} e_h
$$
然后求梯度的时候统一对 $e_h$ 求梯度，更新。

#### 思路2：分数的相加

第二种思路是，我们不同模块保持不同的 embedding，并不把它们在 gradient 层面相加，而是在提议分数层面相加。
我们分别在不同的空间计算提议的 score，然后加起来。其合理性在于，无论有多少个 embedding 空间，其候选的 token 是同一个离散集合 $[1,K]$ 的同一个元素。

在 decoder 空间：
$$
s^d(k) = \frac{1}{2}\nabla_{e^d_{h_i}} d_\theta^\top(e^d_{h'_i}-e^d_{h_i})+\frac{1}{2\alpha^d}\Vert e^d_{h'_i}-e^d_{h_i}\Vert_p^p
$$
在 prior 空间：
$$
s^f(k) = \frac{1}{2}\nabla_{e^f_{h_i}} f_\psi^\top(e^f_{h'_i}-e^f_{h_i})+\frac{1}{2\alpha^f}\Vert e^f_{h'_i}-e^f_{h_i}\Vert_p^p
$$
> 在 prior 空间，我们这里只写了一个 embedding 的，事实上，如果 input embedding 和 output embedding 都存在且不同，则需要分别计算 $s^{f,\text{in}}(k)$, $s^{f,\text{out}}(k)$ ，相加，但总之都是在 prior 空间。而按照 GPT-2 的设置，使用了 weight tying，则 input embedding 和 output embedding 是同一个，可以简化为上面的式子。

总的提议得分为：
$$
s(k) = s^d(k)+s^f(k)
$$
然后提议分布写为：
$$
q(k)= \frac{\exp \left(-s(k)\right)}{\sum_{j=1}^K \exp \left(-s(j)\right)}
$$
就可以进行完整的 NCG 采样。

#### Summary

目前更加倾向于使用思路 2，因为两个不同部分的 embedding 的语义完全不同，distortion model 部分的 embedding，是为了重建的质量，而 prior model 部分的 embedding 是为了拟合先验。思路 1 在底层使用相同的 embedding，然后只经过简单地线性映射到不同的空间，有可能造成表达能力不足。而且共用一个 embedding 也会造成两个部分模块的耦合，不利于分别管理，写代码时比较复杂。

思路 2 可以很好地分别处理两部分内部的更新，然后汇总。各自可以管理各自的 embedding，计算梯度。而且很重要的一点，两部分的码本大小可能并不是完全相同的，distortion 的码本大小就是 $K$，而 prior 的码本中，除了有效码向量外，需要额外引入一个 special token `BOS`，表示一个开始。如果使用思路1，需要我们底层的 embedding 也维护一个 $K+1$ 大小的码本，在 distortion 部分还需要删去这一个特殊token，比较麻烦。

## LD Sampler

朗之万采样是对连续空间的采样方法，在 JSA+perceptual loss 中主要用于第三阶段对 $x$ 的采样。

联合分布为：
$$
p_\Theta(x, h) \propto \exp(-u_\theta(x)) = \exp(-d_\theta(x, h) - f_\psi(h))
$$

则 LD 的更新规则为：
$$
x_{t+1} = x_t - \frac{\alpha^2}{2} \nabla_x d_\theta(x_t, h) + \alpha \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$
其中我们只对 $d_\theta$ 的部分进行更新，因为 $f_\psi(h)$ 不依赖于 $x$，所以其梯度为 0。

这是更新规则，实际操作中，我们也可以加入一个 Metropolis-Hastings 步骤来接受或拒绝这个提议，以确保采样的正确性。接受概率记为 $\rho$：
$$\rho = \min\left(1, \frac{\pi_\theta(x')}{\pi_\theta(x)} \cdot \frac{q(x\mid x')}{q(x'\mid x)}\right)$$
其中 $x$ 是当前的样本，$x'$ 是提议得到的样本。则 $\pi_\theta(x) = p_\Theta(x, h)\propto \exp(-d_\theta(x, h))$， $q(x'\mid x)$ 是 LD 的提议分布，即 $q(x'\mid x) = \mathcal{N}(x' \mid x - \frac{\alpha^2}{2} \nabla_x d_\theta(x, h), \alpha^2 I)$。

则接受概率可以写为：
$$
\rho = \min\left(1, \exp(-d_\theta(x', h) + d_\theta(x, h)) \cdot \frac{\mathcal{N}(x \mid x' - \frac{\alpha^2}{2} \nabla_x d_\theta(x', h), \alpha^2 I)}{\mathcal{N}(x' \mid x - \frac{\alpha^2}{2} \nabla_x d_\theta(x, h), \alpha^2 I)}\right)
$$
然后我们可以接受或拒绝这个样本 $x'$，以此来保证采样的正确性。
