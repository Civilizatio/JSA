# sripts/infer.py
import torch
from src.models.jsa import JSA
from torch.utils.data import DataLoader
from src.data.mnist import MNISTDataset
import math
import numpy as np

# 加载模型
exp_dir = "egs/continuous_mnist/categorical_prior/version_8"
config_path = f"{exp_dir}/config.yaml"
checkpoint_path = f"{exp_dir}/checkpoints/best-checkpoint.ckpt"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = JSA.load_model(
    config_path=config_path, checkpoint_path=checkpoint_path, device=device
)

# 准备测试数据
test_dataset = MNISTDataset(root="./data", train=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

num_latent_vars = model.proposal_model.num_latent_vars
num_categories = model.proposal_model._num_categories


codebook_size = math.prod(num_categories)
codebook_counter = np.zeros(codebook_size, dtype=np.int32)
print(f"Number of latent variables: {num_latent_vars}, Codebook size: {codebook_size}")

# 遍历测试数据并统计码本利用率
with torch.no_grad():  # 禁用梯度计算
    for batch in test_loader:
        x, _, idx = batch
        x = x.to(device)
        h = model.proposal_model.sample_latent(x, num_samples=1, encoded=False)
        # print("Sampled latent variables h shape:", h.shape)

        h = h.cpu().numpy()  # [B, num_latent_vars]
        indices = h.dot(
            np.array(
                [math.prod(num_categories[i + 1 :]) for i in range(num_latent_vars)]
            )
        )  # 将多维类别索引转换为一维索引
        for index in indices:
            codebook_counter[index] += 1
# 计算码本利用率
used_codewords = np.sum(codebook_counter > 0)
utilization_rate = used_codewords / codebook_size * 100
print(f"Used codewords: {used_codewords}/{codebook_size}")
print(f"Codebook utilization rate: {utilization_rate:.4f}%")

# 绘制一维分布图
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.bar(range(codebook_size), codebook_counter, width=1.0)
plt.xlabel("Codeword Index")
plt.ylabel("Usage Count")
plt.title("Codebook Usage Distribution")
plt.tight_layout()
plt.savefig(f"{exp_dir}/codebook_usage_distribution.png")
plt.close()