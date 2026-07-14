"""
mambavision_unet.py
-------------------
MambaVision encoder + U-Net decoder for underwater image restoration.

Uses the official NVIDIA MambaVision backbone (Hatamizadeh & Kautz, 2024)
loaded via HuggingFace Transformers with ``trust_remote_code=True``.

On systems without CUDA (Windows CPU-only), a built-in pure-PyTorch stub is
automatically injected for the ``selective_scan_fn`` used by the Mamba mixer
blocks.  This keeps the code portable at the cost of slower inference; on a
real CUDA GPU the real ``mamba_ssm`` package takes priority automatically.

Architecture (MambaVision-T, the default)
------------------------------------------
MambaVision has 4 hierarchical levels.  Levels 0-1 are purely convolutional
(ConvBlock); levels 2-3 mix Mamba SSM blocks with Transformer self-attention.
Each ``MambaVisionLayer.forward()`` already returns both the downsampled next
input and the pre-downsampling skip tensor, so no manual splitting is needed.

  patch_embed     (3→64→dim ch,  H/4)
  level 0  →  (x@H/8,  s0@H/4,   dim   ch)
  level 1  →  (x@H/16, s1@H/8,   2×dim ch)
  level 2  →  (x@H/32, s2@H/16,  4×dim ch)
  level 3  →  (x@H/32, bn@H/32,  8×dim ch)  ← bottleneck (no downsample)

Decoder (same lightweight DecoderBlock used by ResNet/MobileNet variants):
  dec3 (8d + 4d → 256)   H/16
  dec2 (256 + 2d → 128)  H/8
  dec1 (128 +  d →  64)  H/4
  dec0 (64      →  32)   H   (no skip)
  head Conv1×1 → Sigmoid → (B, 3, H, W)

For MambaVision-T:  d=80  → dims 80/160/320/640
For MambaVision-S:  d=96  → dims 96/192/384/768
For MambaVision-B:  d=128 → dims 128/256/512/1024

Extra channels for physics ablation
------------------------------------
  in_channels = 3  → RGB only
  in_channels = 4  → RGB + t(x)  OR  RGB + B
  in_channels = 5  → RGB + t(x) + B

The first Conv2d in patch_embed.conv_down is re-initialised for in_channels ≠ 3.
Pretrained RGB weights are preserved in the first 3 channel slots.

Installation
------------
  pip install transformers>=4.40.0 huggingface_hub>=0.20.0 timm einops

  For maximum throughput on GPU (Linux + CUDA + nvcc):
    pip install causal-conv1d mamba-ssm --no-build-isolation
"""

import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DecoderBlock

# ---------------------------------------------------------------------------
# Dimension tables
# ---------------------------------------------------------------------------

_SIZE_TO_BASE_DIM: dict[str, int] = {
    "T": 80,
    "S": 96,
    "B": 128,
}

_SIZE_TO_HF_ID: dict[str, str] = {
    "T": "nvidia/MambaVision-T-1K",
    "S": "nvidia/MambaVision-S-1K",
    "B": "nvidia/MambaVision-B-1K",
}


# ---------------------------------------------------------------------------
# Pure-PyTorch mamba_ssm stub
# ---------------------------------------------------------------------------


def _selective_scan_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
    return_last_state: object = False,
) -> torch.Tensor:
    """
    Reference (pure-PyTorch) selective scan — mathematically equivalent to
    ``selective_scan_fn`` from the ``mamba_ssm`` package but runs on any
    device without CUDA kernel compilation.

    Shapes follow the MambaVisionMixer calling convention:
        u     : (B, d_inner, L)
        delta : (B, d_inner, L)
        A     : (d_inner, d_state)
        B     : (B, d_state, L)
        C     : (B, d_state, L)
        D     : (d_inner,)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    A = A.float()
    B = B.float()
    C = C.float()
    D = D.float()

    if delta_bias is not None:
        delta = delta + delta_bias.float().unsqueeze(-1)
    if delta_softplus:
        delta = F.softplus(delta)

    batch, d_in, L = u.shape
    n = A.shape[1]

    # Discretise A and B  (zero-order hold)
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)

    # Sequential SSM scan (the "slow but correct" path)
    x = torch.zeros(batch, d_in, n, device=u.device, dtype=torch.float32)
    ys = []
    for i in range(L):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]  # (B, d_in, n)
        y = torch.einsum("bdn,bn->bd", x, C[:, :, i])  # (B, d_in)
        ys.append(y)

    y = torch.stack(ys, dim=2)  # (B, d_in, L)
    out = y + u * D.unsqueeze(-1)

    if z is not None:
        out = out * F.silu(z.float())

    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, x)


def _inject_mamba_ssm_stub() -> None:
    """
    If the real ``mamba_ssm`` package is not installed, inject a minimal stub
    into ``sys.modules`` so the HuggingFace modeling code can be imported.
    The stub's ``selective_scan_fn`` is the pure-PyTorch reference scan above.
    """
    # Try the real package first — if it works, we're done.
    try:
        import mamba_ssm  # noqa: F401

        return
    except (ImportError, Exception):
        pass

    # Build minimal stub module hierarchy
    m_root = types.ModuleType("mamba_ssm")
    m_ops = types.ModuleType("mamba_ssm.ops")
    m_ssi = types.ModuleType("mamba_ssm.ops.selective_scan_interface")

    m_ssi.selective_scan_fn = _selective_scan_ref
    m_ssi.selective_scan_ref = _selective_scan_ref

    m_root.ops = m_ops
    m_ops.selective_scan_interface = m_ssi

    sys.modules.setdefault("mamba_ssm", m_root)
    sys.modules.setdefault("mamba_ssm.ops", m_ops)
    sys.modules.setdefault("mamba_ssm.ops.selective_scan_interface", m_ssi)


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------


def _load_hf_backbone(model_size: str, pretrained: bool) -> nn.Module:
    """
    Load ``nvidia/MambaVision-{T,S,B}-1K`` from HuggingFace Hub.

    Injects the pure-PyTorch mamba_ssm stub *before* the transformers import
    so that the model code can be loaded on any device/OS.
    """
    # Must happen before AutoModel triggers the check_imports scan
    _inject_mamba_ssm_stub()

    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' library is required for MambaVisionUNet.\n"
            "Install with:  pip install transformers huggingface_hub timm einops"
        ) from exc

    hf_id = _SIZE_TO_HF_ID[model_size]

    # Workaround for HuggingFace meta-device initialization crashing on torch.linspace
    import torch

    orig_linspace = torch.linspace

    def patched_linspace(*args, **kwargs):
        if kwargs.get("device") is None:
            kwargs["device"] = "cpu"
        return orig_linspace(*args, **kwargs)

    torch.linspace = patched_linspace

    # Workaround for Kaggle transformers version mismatch looking for all_tied_weights_keys
    import transformers

    if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
        transformers.PreTrainedModel.all_tied_weights_keys = property(lambda self: {})

    try:
        if pretrained:
            hf_model = AutoModel.from_pretrained(
                hf_id, trust_remote_code=True, low_cpu_mem_usage=False
            )
        else:
            cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
            hf_model = AutoModel.from_config(cfg, trust_remote_code=True)
    finally:
        torch.linspace = orig_linspace

    return hf_model


def _get_backbone_core(hf_model: nn.Module) -> nn.Module:
    """
    ``AutoModel`` wraps the backbone as ``hf_model.model`` (a ``MambaVision``
    instance that exposes ``patch_embed``, ``levels``, and ``forward_features``).
    """
    backbone_core = getattr(hf_model, "model", None)
    if backbone_core is None or not hasattr(backbone_core, "levels"):
        raise AttributeError(
            f"Expected hf_model.model to be MambaVision (with 'levels'); "
            f"got: {type(backbone_core)}.  "
            f"The HuggingFace model code may have changed."
        )
    return backbone_core


def _patch_first_conv(
    patch_embed: nn.Module,
    in_channels: int,
    pretrained: bool,
) -> None:
    """
    Replace the first Conv2d in ``patch_embed.conv_down`` to accept
    ``in_channels`` channels instead of 3.

    PatchEmbed structure from modeling_mambavision.py::

        self.proj     = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),   ← index 0
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
        )
    """
    conv_down = getattr(patch_embed, "conv_down", None)
    if conv_down is None:
        raise AttributeError(
            "patch_embed has no 'conv_down' attribute. "
            "The HuggingFace model structure may have changed."
        )

    # Find the first Conv2d in the sequential
    first_idx: int | None = None
    first_conv: nn.Conv2d | None = None
    for idx, layer in enumerate(conv_down):
        if isinstance(layer, nn.Conv2d):
            first_idx = idx
            first_conv = layer
            break

    if first_conv is None:
        raise RuntimeError("No Conv2d found in patch_embed.conv_down.")

    new_conv = nn.Conv2d(
        in_channels,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )

    if pretrained:
        with torch.no_grad():
            n = min(in_channels, 3)
            new_conv.weight[:, :n] = first_conv.weight[:, :n]

    conv_down[first_idx] = new_conv


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MambaVisionUNet(nn.Module):
    """
    MambaVision encoder + lightweight U-Net decoder.

    Loads a pretrained MambaVision backbone from HuggingFace Hub via the
    ``transformers`` library.  A built-in pure-PyTorch fallback for the
    Mamba selective-scan is automatically used on CPU/Windows environments
    where the ``mamba_ssm`` CUDA package cannot be compiled.

    Args:
        in_channels  (int):  Input channels – 3, 4, or 5.
        out_channels (int):  Output channels. Default: 3.
        model_size   (str):  MambaVision variant ``"T"`` | ``"S"`` | ``"B"``.
                             Default: ``"T"`` (Tiny, ~31 M params).
        pretrained   (bool): Download ImageNet-1K pretrained weights from the
                             HuggingFace Hub on first instantiation.
                             Default: True.

    Example::

        model = MambaVisionUNet(in_channels=5)
        x = torch.randn(2, 5, 256, 256)
        y = model(x)   # (2, 3, 256, 256)
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 3,
        model_size: str = "T",
        pretrained: bool = True,
    ):
        super().__init__()

        if model_size not in _SIZE_TO_BASE_DIM:
            raise ValueError(
                f"model_size must be one of {list(_SIZE_TO_BASE_DIM)}; got '{model_size}'."
            )

        # ------------------------------------------------------------------
        # Load HuggingFace backbone
        # ------------------------------------------------------------------
        hf_model = _load_hf_backbone(model_size, pretrained)
        self.backbone = hf_model  # MambaVisionModel (HF wrapper)
        self.backbone_core = _get_backbone_core(hf_model)  # MambaVision (backbone)

        # ------------------------------------------------------------------
        # Patch first conv for extra physics channels
        # ------------------------------------------------------------------
        if in_channels != 3:
            _patch_first_conv(self.backbone_core.patch_embed, in_channels, pretrained)

        # ------------------------------------------------------------------
        # Decoder  (in_ch, skip_ch, out_ch)
        # ------------------------------------------------------------------
        d = _SIZE_TO_BASE_DIM[model_size]  # base dim per size
        self.dec3 = DecoderBlock(d * 8, d * 4, 256)  # 640+320→256 for T
        self.dec2 = DecoderBlock(256, d * 2, 128)  # 256+160→128
        self.dec1 = DecoderBlock(128, d * 1, 64)  # 128+ 80→ 64
        self.dec0 = DecoderBlock(64, 0, 32)  # no skip at full res

        # patch_embed does 4× total downsampling (two stride-2 convs), so
        # the shallowest skip s0 is at H/4.  After dec0 we are at H/2.
        # One extra bilinear ×2 restores full resolution before the 1×1 head.
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): (N, C_in, H, W)  C_in ∈ {3, 4, 5}.

        Returns:
            Tensor: (N, 3, H, W) restored RGB image in [0, 1].
        """
        # ---- Encoder ----------------------------------------------------
        # MambaVision.forward_features returns (x_global, [s0, s1, s2, bn])
        # where each sN is the pre-downsample feature at that level:
        #   s0 : (N,   d, H/4,  W/4 )
        #   s1 : (N,  2d, H/8,  W/8 )
        #   s2 : (N,  4d, H/16, W/16)
        #   bn : (N,  8d, H/32, W/32)  ← bottleneck (level 3 has no downsample)
        _, skips = self.backbone_core.forward_features(x)
        s0, s1, s2, bn = skips

        # ---- Decoder ----------------------------------------------------
        d3 = self.dec3(bn, s2)  # (N, 256, H/16, W/16)
        d2 = self.dec2(d3, s1)  # (N, 128, H/8,  W/8 )
        d1 = self.dec1(d2, s0)  # (N,  64, H/4,  W/4 )
        d0 = self.dec0(d1)  # (N,  32, H,    W   )

        return self.head(d0)  # (N,   3, H,    W   )
