# 实验总结

变量：
batch_size 保持一致（小 batch；大 batch）
输出的数据范围是否加限制（无限制；归一化到 $[-1,1]$）
学习率的变化（和 VQ-VAE 保持一致；修改 jsa 的学习率）

JSA：
batch_size:128(4)
无限制

- 学习率设置为 5.86e-07 （等价学习率为 3e-4，是我原来 $450*4$ 的 情况下能跑的）[](egs/cifar10/jsa/categorical_prior_conv/2026-01-14_22-18-02)
- 再设置同 VQ-GAN 一致的 4.5e-06（等价学习率为 2e-3）[](egs/cifar10/jsa/categorical_prior_conv/2026-01-14_22-46-23)
- 再设置 更小的 1.86e-7 （等价学习率为9.5e-5，更小的）[](egs/cifar10/jsa/categorical_prior_conv/2026-01-14_22-53-18)
- 设置为 4.5e-07 （等价学习率为 2e-4）， $\sigma$ 的调整增加 hold steps，减小 mis 步骤到 10 [](egs/cifar10/jsa/categorical_prior_conv/2026-01-16_10-12-30)
- 设置为 4.5e-07，$\sigma$ 的调整增加 hold steps.[](egs/cifar10/jsa/categorical_prior_conv/2026-01-17_11-19-32)

JSA d
batch_size: 128(4)
归一化到 [-1,1]
- 学习率设置为 5.86e-07 （等价学习率为 3e-4，是我原来 $450*4$ 的 情况下能跑的）[](egs/cifar10/jsa/categorical_prior_conv/2026-01-15_15-40-01)
- 再设置同 VQ-GAN 一致的 4.5e-07（等价学习率为 2e-4）[](egs/cifar10/jsa/categorical_prior_conv/2026-01-15_15-41-30)
- 再设置 更小的 1.86e-7 （等价学习率为9.5e-5，更小的）[](egs/cifar10/jsa/categorical_prior_conv/2026-01-15_15-43-39)
- 设置为 4.5e-06（等价学习率为 2e-3）， $\sigma$ 的调整增加 hold steps [](egs/cifar10/jsa/categorical_prior_conv/2026-01-16_10-00-15)
- 设置为 4.5e-06（等价学习率为 2e-3）， $\sigma$ 的调整增加 hold steps， mis 步骤减小到 10 [](egs/cifar10/jsa/categorical_prior_conv/2026-01-16_10-05-44)

- 设置为 4.5e-07，$\sigma$ 的调整增加 hold steps.[](egs/cifar10/jsa/categorical_prior_conv/2026-01-17_11-21-13)
- 设置为 4.5e-07，$\sigma$ 的调整增加 hold steps， mis 步骤减小到 10 [](egs/cifar10/jsa/categorical_prior_conv/2026-01-17_11-22-15)

使用 4.5e-06，训练过程中 nll 最小嗷，梯度也最小，但是很容易造成码本坍塌，利用率很低
而使用 4.5e-07，最后的码本利用率挺大的。

因此，学习率一定不能太大，现在学习率较大的跑的都不好。

需要设置为 4.5e-07，增加 hold steps，减小 mis 步骤到 10 的配置继续跑更长时间看看效果

对 JSA 再跑两个验证实验：

用 4.5e-07 的学习率，改变 encoder/decoder 的网络结构，对比实验结果。
分为两个：
归一化到 [-1,1] 和无限制两种情况。
同时，加一个通道数为 256 和 通道数为 12 的情况的对比
一共四种情况：
1. 归一化到 [-1,1]，通道数 256[](egs/cifar10/jsa/categorical_prior_conv/2026-01-19_16-09-19)
2. 归一化到 [-1,1]，通道数 12[](egs/cifar10/jsa/categorical_prior_conv/2026-01-19_16-20-55)
3. 无限制，通道数 256[](egs/cifar10/jsa/categorical_prior_conv/2026-01-19_16-17-12)
4. 无限制，通道数 12[](egs/cifar10/jsa/categorical_prior_conv/2026-01-19_16-18-24)

补充一个原来可以跑出来的最好的实验
学习率 3e-4，归一化到 [-1,1]，[](egs/cifar10/jsa/categorical_prior_conv/2026-01-20_15-49-05)
以及与其对应的网络结构略有不同的通道数为 256 的情况
[](egs/cifar10/jsa/categorical_prior_conv/2026-01-20_15-55-07)


VQ-GAN：
batch_size: 128(4)

无限制
- 学习率设置为 4.5e-06，等价学习率为 2e-3 [](egs/cifar10/vqgan/vq_gan_cifar10/2026-01-19_00-08-24)
- 设置为 4.5e-06，加上 perceptual loss [](egs/cifar10/vqgan/vq_gan_cifar10/2026-01-19_00-09-59)

归一化到 [-1,1]
- 学习率设置为 4.5e-06，等价学习率为 2e-3 [](egs/cifar10/vqgan/vq_gan_cifar10/2026-01-19_10-59-58)
- 设置为 4.5e-06，加上 perceptual loss [](egs/cifar10/vqgan/vq_gan_cifar10/2026-01-19_11-01-50)

## 统计码本的分布情况

这里主要想要证明的是，对于不同类别，不同位置的 codeword， 我们建模为独立同分布的 categorical distribution 是不合理的。

因此，我们需要统计每个 codeword 在不同类别，不同位置上的使用频率。

一开始画出不同类别以及不同位置的 codeword 使用频率的直方图，但由于码本大小 1024，不便于观察。

后面改为统计一些分布上的特征：

1. codebook 分布之间的距离矩阵。由于我们希望距离是对称的，这里使用 JS 散度来衡量。
    JS 散度：
    $$JS(P||Q) = 0.5 * (KL(P||M) + KL(Q||M))$$
    其中 $M = 0.5 * (P + Q)$
2. codebook 使用的 top-k support overlap。即对于每个 codebook 分布，取其 top-k 的 codeword，计算不同 codebook 之间 top-k codeword 的重合度。
   重合度定义为：
   $$overlap(P, Q) = \frac{|topk(P) \cap topk(Q)|}{k}$$
3. 每个 codebook 分布视为一个 1024 维的向量，通过 PCA/UMAP 降维到 2 维进行可视化，观察不同类别、不同位置的 codebook 分布的聚类情况。

如果这些 codebook 分布是独立同分布的，那么我们期望看到：
1. 距离矩阵中不同类别、不同位置的 codebook 之间的距离应该比较接近，没有明显的聚类现象。
2. top-k support overlap 应该比较高，说明不同 codebook 之间有较多重合的 codeword。
3. 降维后的可视化图中，不同类别、不同位置的 codebook 分布应该混杂在一起，没有明显的聚类现象。

最后，我们可以使用 top-k 的 codeword 来进行图像的重构。
