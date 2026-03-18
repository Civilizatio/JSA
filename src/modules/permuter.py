# src/modules/permuter.py
""" This module defines various permutation strategies for rearranging the order of pixels in an image.
Each permutation is implemented as a subclass of AbstractPermuter, which defines a common interface for forward and reverse transformations.
The module also includes a visualization utility to plot the scan paths of different permuters.
"""

import torch
import torch.nn as nn
import numpy as np

from src.base.base_permuter import AbstractPermuter


class Identity(AbstractPermuter):
    def __init__(self):
        super().__init__()

    def forward(self, x, reverse=False):
        return x


class Subsample(AbstractPermuter):
    def __init__(self, H, W):
        super().__init__()
        C = 1
        indices = np.arange(H * W).reshape(C, H, W)
        while min(H, W) > 1:
            indices = indices.reshape(C, H // 2, 2, W // 2, 2)
            indices = indices.transpose(0, 2, 4, 1, 3)
            indices = indices.reshape(C * 4, H // 2, W // 2)
            H = H // 2
            W = W // 2
            C = C * 4
        assert H == W == 1
        idx = torch.tensor(indices.ravel())
        self.register_buffer(
            "forward_shuffle_idx", nn.Parameter(idx, requires_grad=False)
        )
        self.register_buffer(
            "backward_shuffle_idx",
            nn.Parameter(torch.argsort(idx), requires_grad=False),
        )

    def forward(self, x, reverse=False):
        if not reverse:
            return x[:, self.forward_shuffle_idx]
        else:
            return x[:, self.backward_shuffle_idx]


def mortonify(i, j):
    """(i,j) index to linear morton code"""
    i = np.uint64(i)
    j = np.uint64(j)

    z = np.uint(0)

    for pos in range(32):
        z = (
            z
            | ((j & (np.uint64(1) << np.uint64(pos))) << np.uint64(pos))
            | ((i & (np.uint64(1) << np.uint64(pos))) << np.uint64(pos + 1))
        )
    return z


class ZCurve(AbstractPermuter):
    def __init__(self, H, W):
        super().__init__()
        reverseidx = [np.int64(mortonify(i, j)) for i in range(H) for j in range(W)]
        idx = np.argsort(reverseidx)
        idx = torch.tensor(idx)
        reverseidx = torch.tensor(reverseidx)
        self.register_buffer("forward_shuffle_idx", idx)
        self.register_buffer("backward_shuffle_idx", reverseidx)

    def forward(self, x, reverse=False):
        if not reverse:
            return x[:, self.forward_shuffle_idx]
        else:
            return x[:, self.backward_shuffle_idx]


class SpiralOut(AbstractPermuter):
    def __init__(self, H, W):
        super().__init__()
        assert H == W
        size = W
        indices = np.arange(size * size).reshape(size, size)

        i0 = size // 2
        j0 = size // 2 - 1

        i = i0
        j = j0

        idx = [indices[i0, j0]]
        step_mult = 0
        for c in range(1, size // 2 + 1):
            step_mult += 1
            # steps left
            for k in range(step_mult):
                i = i - 1
                j = j
                idx.append(indices[i, j])

            # step down
            for k in range(step_mult):
                i = i
                j = j + 1
                idx.append(indices[i, j])

            step_mult += 1
            if c < size // 2:
                # step right
                for k in range(step_mult):
                    i = i + 1
                    j = j
                    idx.append(indices[i, j])

                # step up
                for k in range(step_mult):
                    i = i
                    j = j - 1
                    idx.append(indices[i, j])
            else:
                # end reached
                for k in range(step_mult - 1):
                    i = i + 1
                    idx.append(indices[i, j])

        assert len(idx) == size * size
        idx = torch.tensor(idx)
        self.register_buffer("forward_shuffle_idx", idx)
        self.register_buffer("backward_shuffle_idx", torch.argsort(idx))

    def forward(self, x, reverse=False):
        if not reverse:
            return x[:, self.forward_shuffle_idx]
        else:
            return x[:, self.backward_shuffle_idx]


class SpiralIn(AbstractPermuter):
    def __init__(self, H, W):
        super().__init__()
        assert H == W
        size = W
        indices = np.arange(size * size).reshape(size, size)

        i0 = size // 2
        j0 = size // 2 - 1

        i = i0
        j = j0

        idx = [indices[i0, j0]]
        step_mult = 0
        for c in range(1, size // 2 + 1):
            step_mult += 1
            # steps left
            for k in range(step_mult):
                i = i - 1
                j = j
                idx.append(indices[i, j])

            # step down
            for k in range(step_mult):
                i = i
                j = j + 1
                idx.append(indices[i, j])

            step_mult += 1
            if c < size // 2:
                # step right
                for k in range(step_mult):
                    i = i + 1
                    j = j
                    idx.append(indices[i, j])

                # step up
                for k in range(step_mult):
                    i = i
                    j = j - 1
                    idx.append(indices[i, j])
            else:
                # end reached
                for k in range(step_mult - 1):
                    i = i + 1
                    idx.append(indices[i, j])

        assert len(idx) == size * size
        idx = idx[::-1]
        idx = torch.tensor(idx)
        self.register_buffer("forward_shuffle_idx", idx)
        self.register_buffer("backward_shuffle_idx", torch.argsort(idx))

    def forward(self, x, reverse=False):
        if not reverse:
            return x[:, self.forward_shuffle_idx]
        else:
            return x[:, self.backward_shuffle_idx]


class Random(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        indices = np.random.RandomState(1).permutation(H * W)
        idx = torch.tensor(indices.ravel())
        self.register_buffer("forward_shuffle_idx", idx)
        self.register_buffer("backward_shuffle_idx", torch.argsort(idx))

    def forward(self, x, reverse=False):
        if not reverse:
            return x[:, self.forward_shuffle_idx]
        else:
            return x[:, self.backward_shuffle_idx]


class AlternateParsing(AbstractPermuter):
    def __init__(self, H, W):
        super().__init__()
        indices = np.arange(W * H).reshape(H, W)
        for i in range(1, H, 2):
            indices[i, :] = indices[i, ::-1]
        idx = indices.flatten()
        assert len(idx) == H * W
        idx = torch.tensor(idx)
        self.register_buffer("forward_shuffle_idx", idx)
        self.register_buffer("backward_shuffle_idx", torch.argsort(idx))

    def forward(self, x, reverse=False):
        if not reverse:
            return x[:, self.forward_shuffle_idx]
        else:
            return x[:, self.backward_shuffle_idx]


# ======= Visualization Utility =======

import matplotlib.pyplot as plt
import os
from matplotlib.collections import LineCollection


def plot_permuter(name, permuter_cls, H=8, W=8):
    try:
        # For Identity permuter, no indices to retrieve
        if permuter_cls is Identity:
            permuter = permuter_cls()
            path_indices = np.arange(H * W)
        else:
            permuter = permuter_cls(H, W)
            # Try to get the forward shuffle indices
            if hasattr(permuter, "forward_shuffle_idx"):
                path_indices = permuter.forward_shuffle_idx.cpu().numpy().ravel()
            else:
                print(f"Skipping {name}: Cannot retrieve shuffle indices directly.")
                return
    except Exception as e:
        print(f"Skipping {name}: {e}")
        return

    ys, xs = np.unravel_index(path_indices, (H, W))

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # --- 1. Gradient scan path plot ---
    ax[0].set_title(f"{name}: Gradient Scan Path (Blue->Red)")

    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Use LineCollection to create gradient colored lines
    # cmap='jet': Blue(Start) -> Green(Middle) -> Red(End), high contrast, easy to see order
    norm = plt.Normalize(0, len(xs))
    lc = LineCollection(segments, cmap="jet", norm=norm, alpha=0.7)
    lc.set_array(np.arange(len(xs)))  # Color is based on step index
    lc.set_linewidth(2)

    line = ax[0].add_collection(lc)

    # Add arrows to indicate direction (draw every few steps to avoid clutter)
    step = max(1, (H * W) // 12)
    for i in range(0, len(xs) - 1, step):
        dx = xs[i + 1] - xs[i]
        dy = ys[i + 1] - ys[i]
        # Draw arrow only if there is movement
        if dx != 0 or dy != 0:
            ax[0].arrow(
                xs[i],
                ys[i],
                dx * 0.4,
                dy * 0.4,
                shape="full",
                lw=0,
                length_includes_head=True,
                head_width=0.25,
                color="black",
                alpha=0.6,
            )

    # Mark start and end points
    ax[0].plot(
        xs[0], ys[0], "go", markersize=12, label="Start", zorder=10, markeredgecolor="k"
    )
    ax[0].plot(
        xs[-1], ys[-1], "rX", markersize=12, label="End", zorder=10, markeredgecolor="k"
    )

    # Label the first few points with their indices
    for i in range(min(10, H * W)):
        ax[0].text(
            xs[i],
            ys[i],
            str(i),
            fontsize=9,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            bbox=dict(boxstyle="circle,pad=0.1", fc="black", alpha=0.6),
        )

    ax[0].set_xlim(-0.5, W - 0.5)
    ax[0].set_ylim(H - 0.5, -0.5)  # Image coordinate system, y axis downward
    ax[0].grid(True, linestyle="--", alpha=0.3)
    ax[0].set_xticks(range(W))
    ax[0].set_yticks(range(H))
    ax[0].legend()
    # Colorbar for path
    cb1 = fig.colorbar(line, ax=ax[0], fraction=0.046, pad=0.04)
    cb1.set_label("Step Index")

    # --- 2. Order heatmap (using the same colormap) ---
    order_map = np.zeros((H, W))
    for t, (y, x) in enumerate(zip(ys, xs)):
        order_map[y, x] = t

    ax[1].set_title(f"{name}: Order Heatmap")
    im = ax[1].imshow(order_map, cmap="jet")
    for y in range(H):
        for x in range(W):
            # Adjust text color: use white if background is too dark or too bright, otherwise black
            val = order_map[y, x]
            limit = H * W
            text_color = "white" if val < limit * 0.3 or val > limit * 0.7 else "black"

            ax[1].text(
                x, y, int(val), ha="center", va="center", color=text_color, fontsize=8
            )

    cb2 = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    cb2.set_label("Sequence Step")

    plt.tight_layout()
    filename = f"permuter_{name.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"Saved visualization to {os.path.abspath(filename)}")
    plt.close(fig)  # Close figure to free memory


if __name__ == "__main__":
    p0 = AlternateParsing(16, 16)
    print(p0.forward_shuffle_idx)
    print(p0.backward_shuffle_idx)

    x = torch.randint(0, 768, size=(11, 256))
    y = p0(x)
    xre = p0(y, reverse=True)
    assert torch.equal(x, xre)

    p1 = SpiralOut(2, 2)
    print(p1.forward_shuffle_idx)
    print(p1.backward_shuffle_idx)

    permuters = {
        "Identity": Identity,
        "Subsample": Subsample,
        "ZCurve": ZCurve,
        "SpiralOut": SpiralOut,
        "SpiralIn": SpiralIn,
        "Random": Random,
        "AlternateParsing": AlternateParsing,
    }
    print("Plotting permuter scan paths...")
    plt.switch_backend("Agg")
    for name, cls in permuters.items():
        plot_permuter(name, cls, H=8, W=8)
