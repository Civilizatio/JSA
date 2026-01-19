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


VQ-GAN：
batch_size: 128(4)

无限制
- 学习率设置为 4.5e-06，等价学习率为 2e-3 [](egs/cifar10/vqgan/vq_gan_cifar10/2026-01-19_00-08-24)
- 设置为 4.5e-06，加上 perceptual loss [](egs/cifar10/vqgan/vq_gan_cifar10/2026-01-19_00-09-59)

归一化到 [-1,1]
- 学习率设置为 4.5e-06，等价学习率为 2e-3 [](egs/cifar10/vqgan/vq_gan_cifar10/2026-01-19_10-59-58)
- 设置为 4.5e-06，加上 perceptual loss [](egs/cifar10/vqgan/vq_gan_cifar10/2026-01-19_11-01-50)


