"""Multi-camera ACT with a pretrained spatial backbone and a masked CVAE posterior."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


@dataclass
class ACTConfig:
    state_dim: int
    action_dim: int
    image_size: int = 160
    chunk_size: int = 32
    hidden_dim: int = 256
    latent_dim: int = 32
    heads: int = 8
    encoder_layers: int = 3
    decoder_layers: int = 3
    dropout: float = 0.1
    joint_residual: bool = False

    def __post_init__(self):
        if min(self.state_dim, self.action_dim, self.chunk_size, self.hidden_dim) <= 0:
            raise ValueError("ACT dimensions must be positive")
        if self.hidden_dim % self.heads or self.image_size < 32:
            raise ValueError("Invalid hidden/image size")


class ACT(nn.Module):
    def __init__(self, config: ACTConfig, pretrained: bool = True, backbone_path=None):
        super().__init__()
        self.config = config
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained and backbone_path is None else None)
        if backbone_path is not None:
            backbone.load_state_dict(torch.load(backbone_path, map_location="cpu", weights_only=True))
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.image_projection = nn.Conv2d(512, config.hidden_dim, 1)
        grid = (config.image_size + 31) // 32
        self.spatial_position = nn.Parameter(torch.randn(1, 1, grid * grid, config.hidden_dim) * 0.02)
        self.camera_position = nn.Parameter(torch.randn(1, 3, 1, config.hidden_dim) * 0.02)
        self.state_projection = nn.Linear(config.state_dim, config.hidden_dim)
        self.latent_projection = nn.Linear(config.latent_dim, config.hidden_dim)
        self.action_projection = nn.Linear(config.action_dim, config.hidden_dim)
        self.posterior_cls = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.posterior_position = nn.Parameter(torch.randn(1, config.chunk_size + 2, config.hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(config.hidden_dim, config.heads, 4*config.hidden_dim,
                                          config.dropout, "gelu", batch_first=True, norm_first=True)
        self.posterior = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(config.hidden_dim), enable_nested_tensor=False)
        self.posterior_head = nn.Linear(config.hidden_dim, 2*config.latent_dim)
        self.encoder = nn.TransformerEncoder(layer, config.encoder_layers,
                                            norm=nn.LayerNorm(config.hidden_dim), enable_nested_tensor=False)
        decoder_layer = nn.TransformerDecoderLayer(config.hidden_dim, config.heads, 4*config.hidden_dim,
                                                   config.dropout, "gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers, norm=nn.LayerNorm(config.hidden_dim))
        self.queries = nn.Parameter(torch.randn(1, config.chunk_size, config.hidden_dim) * 0.02)
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)
        if config.joint_residual:
            self.register_buffer('joint_anchor_weight', torch.zeros(config.action_dim, config.state_dim))
            self.register_buffer('joint_anchor_bias', torch.zeros(config.action_dim))
            nn.init.zeros_(self.action_head.weight)
            nn.init.zeros_(self.action_head.bias)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def configure_joint_anchor(self, stats):
        """Express measured joints in normalized action coordinates; predict corrections.

        Only contemporaneous measured joints enter this anchor. Future action
        labels, measured gripper angles and absent spine feedback do not.
        """
        if not self.config.joint_residual:
            return
        from .schema import JOINT_ACTION_INDICES, JOINT_STATE_INDICES
        with torch.no_grad():
            self.joint_anchor_weight.zero_(); self.joint_anchor_bias.zero_()
            for ai, si in zip(JOINT_ACTION_INDICES, JOINT_STATE_INDICES):
                self.joint_anchor_weight[ai, si] = stats['state_std'][si]/stats['action_std'][ai]
                self.joint_anchor_bias[ai] = (stats['state_mean'][si]-stats['action_mean'][ai])/stats['action_std'][ai]

    def train(self, mode=True):
        super().train(mode)
        # Keep ImageNet running moments; three correlated robot views are not an IID image batch.
        for layer in self.backbone.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        return self

    def forward(self, images, state, actions=None, mask=None):
        b, cameras, channels, height, width = images.shape
        if (cameras, channels, height, width) != (3, 3, self.config.image_size, self.config.image_size):
            raise ValueError("Images do not match checkpoint camera/spatial schema")
        images = images.reshape(b*cameras, channels, height, width).float()
        images = (images / 255 - self.image_mean) / self.image_std
        features = self.image_projection(self.backbone(images)).flatten(2).transpose(1, 2)
        features = features.reshape(b, cameras, -1, self.config.hidden_dim)
        features = (features + self.spatial_position + self.camera_position).flatten(1, 2)
        state_token = self.state_projection(state).unsqueeze(1)
        mu = state.new_zeros(b, self.config.latent_dim)
        logvar = torch.zeros_like(mu)
        if actions is not None:
            if mask is None or mask.shape != actions.shape[:2]:
                raise ValueError("Training posterior requires an explicit episode padding mask")
            inputs = torch.cat([self.posterior_cls.expand(b, -1, -1), state_token,
                                self.action_projection(actions.masked_fill(~mask[..., None], 0))], 1)
            padding = torch.cat([torch.zeros(b, 2, device=mask.device, dtype=torch.bool), ~mask], 1)
            hidden = self.posterior(inputs + self.posterior_position, src_key_padding_mask=padding)[:, 0]
            mu, logvar = self.posterior_head(hidden).chunk(2, -1)
            logvar = logvar.clamp(-8, 8)
            latent = mu + torch.randn_like(mu) * (0.5*logvar).exp()
        else:
            latent = mu
        memory = self.encoder(torch.cat([state_token, self.latent_projection(latent).unsqueeze(1), features], 1))
        actions_out = self.action_head(self.decoder(self.queries.expand(b, -1, -1), memory))
        if self.config.joint_residual:
            # Keep the affine coordinate transform in FP32 even during BF16
            # training, to avoid quantizing a small correction on a large joint.
            with torch.autocast(device_type=state.device.type, enabled=False):
                anchor = nn.functional.linear(state.float(), self.joint_anchor_weight.float(), self.joint_anchor_bias.float())
                actions_out = actions_out.float() + anchor[:, None]
        return actions_out, mu, logvar


def loss_fn(prediction, target, mask, mu, logvar, kl_weight=1.0):
    weights = mask[..., None].expand_as(prediction)
    reconstruction = (torch.abs(prediction.float()-target) * weights).sum() / weights.sum().clamp_min(1)
    kl = (-0.5 * (1 + logvar.float() - mu.float().square() - logvar.float().exp())).sum(-1).mean()
    return reconstruction + kl_weight*kl, reconstruction, kl
