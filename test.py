import torch

# 加载 checkpoint
checkpoint_path = "egs/bernoulli_mnist/binary_mnist/version_0/checkpoints/best-checkpoint.ckpt"
checkpoint = torch.load(checkpoint_path)

# 查看 checkpoint 的键
print(checkpoint.keys())

# 查看模型的 state_dict
print(checkpoint["state_dict"].keys())

# 查看自定义的 sampler_state
if "sampler_state" in checkpoint:
    print(checkpoint["sampler_state"])