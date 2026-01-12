import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.losses.lpips import LPIPS_CIFAR10
from src.modules.discriminator import NLayerDiscriminator


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_ndf=64,
        disc_loss="hinge",
        pixel_loss="l1",
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        assert pixel_loss in ["l1", "l2"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS_CIFAR10().eval()
        self.perceptual_weight = perceptual_weight
        self.pixel_loss = pixel_loss

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm,
            ndf=disc_ndf,
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        codebook_loss,
        inputs,
        reconstructions,
        global_step,
        last_layer=None,
        cond=None,
        split="train",
    ):
        # L1 and perceptual loss
        if self.pixel_loss == "l1":
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        elif self.pixel_loss == "l2":
            rec_loss = (inputs.contiguous() - reconstructions.contiguous()) ** 2
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous()
            )
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0], device=rec_loss.device)

        nll_loss = torch.mean(rec_loss)

        # --- GENERATOR LOSS ---
        if cond is None:
            assert not self.disc_conditional
            logits_fake = self.discriminator(reconstructions.contiguous())
        else:
            assert self.disc_conditional
            logits_fake = self.discriminator(
                torch.cat((reconstructions.contiguous(), cond), dim=1)
            )
        g_loss = -torch.mean(logits_fake)

        try:
            d_weight = self.calculate_adaptive_weight(
                nll_loss, g_loss, last_layer=last_layer
            )
        except RuntimeError:
            assert not self.training
            d_weight = torch.tensor(0.0, device=inputs.device)

        disc_factor = adopt_weight(
            self.disc_factor, global_step, threshold=self.discriminator_iter_start
        )

        # Combine all losses for the autoencoder/generator
        aeloss = (
            nll_loss
            + d_weight * disc_factor * g_loss
            + self.codebook_weight * codebook_loss.mean()
        )

        log_ae = {
            f"{split}/aeloss": aeloss.clone().detach().mean(),
            f"{split}/quant_loss": codebook_loss.detach().mean(),
            f"{split}/nll_loss": nll_loss.detach().mean(),
            f"{split}/rec_loss": rec_loss.detach().mean(),
            f"{split}/p_loss": p_loss.detach().mean(),
            f"{split}/d_weight": d_weight.detach(),
            f"{split}/disc_factor": torch.tensor(disc_factor).to(rec_loss.device),
            f"{split}/g_loss": g_loss.detach().mean(),
        }

        # --- DISCRIMINATOR LOSS ---
        if cond is None:
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
        else:
            logits_real = self.discriminator(
                torch.cat((inputs.contiguous().detach(), cond), dim=1)
            )
            logits_fake = self.discriminator(
                torch.cat((reconstructions.contiguous().detach(), cond), dim=1)
            )

        discloss = disc_factor * self.disc_loss(logits_real, logits_fake)

        log_disc = {
            f"{split}/discloss": discloss.clone().detach().mean(),
            f"{split}/logits_real": logits_real.detach().mean(),
            f"{split}/logits_fake": logits_fake.detach().mean(),
        }

        return aeloss, discloss, log_ae, log_disc
