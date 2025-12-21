# src/models/components/losses.py
# from taming-transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components.discriminator import NLayerDiscriminator

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def adopt_weight(weight, global_step, threshold=0, value=0.0):
    """ Adopt a weight value based on the global step.
    
    If global_step < threshold, return value; else return weight.
    Here, value is typically 0.0 to disable a loss term initially.
    Namely, before reaching 'threshold' steps, the gan loss is not applied.
    
    """
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


class JSAGANLoss(nn.Module):
    """
    Simplified GAN Loss module for JSA.
    Handles:
    1. Discriminator Network (NLayerDiscriminator)
    2. Discriminator Loss (Hinge or Vanilla)
    3. Generator Adversarial Loss (with Adaptive Weighting)
    """
    def __init__(
        self,
        disc_start, # step to start discriminator updates
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0, # for time scheduling the disc loss
        disc_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_ndf=64,
        disc_loss="hinge",
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        
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
            
        print(f"JSAGANLoss running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """ Adaptive weight for the adversarial loss, based on the ratio of gradients.
        
        From Taming Transformers:
            lambda = weight * ||∇_x L_NLL|| / (||∇_x L_GAN|| + 1e-4)
        1e-4 is added to the denominator to prevent division by zero.
        The weight is then clamped to [0, 1e4] and scaled by discriminator_weight.
        """
        
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            # Fallback or raise error if last_layer is not provided
            return torch.tensor(1.0, device=nll_loss.device)

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None, # Conditional information for discriminator (if any)
        nll_loss=None, # Required for adaptive weight calculation in generator step
        split="train",
    ):
        # optimizer_idx 0: Generator (JSA) update
        # optimizer_idx 1: Discriminator update

        if optimizer_idx == 0:
            # Generator update: Fool the discriminator
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous(), cond), dim=1)
                )
            
            # Generator wants logits_fake to be classified as real (positive)
            # L_G = - E[logits_fake]
            g_loss = -torch.mean(logits_fake) 

            try:
                # Calculate adaptive weight if nll_loss (reconstruction loss) is provided
                if nll_loss is not None and last_layer is not None:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                else:
                    d_weight = torch.tensor(self.discriminator_weight, device=inputs.device)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0, device=inputs.device)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            
            # The final adversarial loss term to be added to JSA loss
            loss = d_weight * disc_factor * g_loss

            log = {
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach().mean(),
                "{}/adv_loss".format(split): loss.detach().mean(),
            }
            return loss, log

        if optimizer_idx == 1:
            # Discriminator update: Distinguish real from fake
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

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean(),
            }
            return d_loss, log