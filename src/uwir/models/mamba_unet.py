"""
mamba_unet.py
-------------
Native Mamba U-Net for underwater image restoration.

Unlike ``MambaVisionUNet`` (classification backbone bolted onto a decoder),
this model is purpose-built for dense prediction: 2-D Selective Scan (SS2D)
blocks appear at *every* encoder and decoder stage, giving global long-range
context throughout the full spatial hierarchy.

Design choices (all transparent — see implementation_plan.md for rationale):
  dims          (96, 192, 384, 768)  — VMamba-T width; ~24 M total params
  enc_depths    (2, 2, 6, 2)         — heavy bottleneck, light early stages
  dec_depths    (2, 2, 2, 2)         — lighter decoder (refinement, not global modelling)
  patch_embed   stride-4             — first VSSBlocks at H/4 → max L = 4 096 at 256 px
  scan          chunked (chunk=64)   — 64 Python-loop iters vs. 4 096 for naive scan;
                                       each chunk is fully vectorised on CUDA
  4 directions  raster, flip-H, flip-W, transpose — VMamba-style SS2D
  d_state = 16, dt_rank = ceil(dim/16) — standard Mamba defaults
  in_channels ∈ {3, 4, 5}           — physics ablation variants supported
  no pretrained weights              — self-contained, trains from scratch

Scan spatial layout (256 × 256 input)
--------------------------------------
  patch_embed (stride-4)  → H/4  × W/4  = 64 × 64  (L = 4 096)
  enc1  [VSSBlock × 2]    @ H/4  × W/4
  down1 (stride-2)        → H/8  × W/8  = 32 × 32  (L = 1 024)
  enc2  [VSSBlock × 2]    @ H/8  × W/8
  down2 (stride-2)        → H/16 × W/16 = 16 × 16  (L = 256)
  enc3  [VSSBlock × 6]    @ H/16 × W/16
  down3 (stride-2)        → H/32 × W/32 = 8  × 8   (L = 64)
  bn    [VSSBlock × 2]    @ H/32 × W/32
  up3 + dec3 [VSSBlock × 2] → H/16
  up2 + dec2 [VSSBlock × 2] → H/8
  up1 + dec1 [VSSBlock × 2] → H/4
  head (bilinear ×4 + Conv) → H × W

Dimensions (VMamba-T): 96 / 192 / 384 / 768
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# mamba-ssm CUDA kernel (Linux/CUDA only — install with: pip install mamba-ssm)
# On Windows / CPU-only machines the pure-Python chunked scan is used instead.
# ---------------------------------------------------------------------------
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _mamba_selective_scan

    _MAMBA_CUDA = True
except ImportError:
    _MAMBA_CUDA = False


# ---------------------------------------------------------------------------
# Chunked vectorised 2-D selective scan
# ---------------------------------------------------------------------------


def _ssm_scan_chunked(
    u: torch.Tensor,  # (B, d_inner, L)
    delta: torch.Tensor,  # (B, d_inner, L)
    A: torch.Tensor,  # (d_inner, d_state)   negative reals
    B: torch.Tensor,  # (B, d_state, L)
    C: torch.Tensor,  # (B, d_state, L)
    D: torch.Tensor,  # (d_inner,)
    delta_bias: torch.Tensor | None = None,  # (d_inner,)
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Chunked prefix-product SSM scan.

    Memory-efficient design: deltaA and deltaB_u are computed *inside*
    each chunk loop iteration, so peak memory is O(chunk_size) not O(L).

    At enc1 (B=16, d_inner=192, L=4096, d_state=16):
      Naïve pre-compute: (16, 192, 4096, 16) × 2 ≈ 1.6 GB just for those tensors.
      Chunked (chunk=64): (16, 192, 64, 16) × 2 ≈ 25 MB — 64× less.

    Returns: y  (B, d_inner, L)
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
    delta = F.softplus(delta)

    B_sz, d_in, L = u.shape

    ys = []
    x_state = u.new_zeros(B_sz, d_in, A.shape[1])  # (B, d_in, n)

    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)

        u_c = u[:, :, start:end]  # (B, d_in, T)
        delta_c = delta[:, :, start:end]  # (B, d_in, T)
        B_c = B[:, :, start:end]  # (B, n,    T)
        C_c = C[:, :, start:end]  # (B, n,    T)

        # Discretise only this chunk — (B, d_in, T, n), T ≤ chunk_size
        dA = torch.exp(torch.einsum("bdl,dn->bdln", delta_c, A))
        dBu = torch.einsum("bdl,bnl,bdl->bdln", delta_c, B_c, u_c)

        # Parallel prefix product within the chunk
        A_pfx = torch.cumprod(dA, dim=2)  # (B, d_in, T, n)

        # Vectorised recurrence (zero initial state) + carried-over state.
        # Clamp the quotient to ±1e15 so cumsum never overflows to inf:
        # if A_pfx underflows to 0 in fp32 (possible after 40+ fast-decay steps)
        # then 0 × inf = NaN; with the clamp, 0 × finite = 0 (correct: state gone).
        A_pfx_safe = A_pfx.clamp(min=1e-30)
        safe_ratio = (dBu / A_pfx_safe).clamp(min=-1e15, max=1e15)
        x_from_zero = A_pfx * torch.cumsum(safe_ratio, dim=2)
        x_chunk = x_from_zero + A_pfx * x_state.unsqueeze(2)

        # Update carry — detach to truncate BPTT at chunk boundaries.
        # Gradients still flow within each chunk (64 tokens). This is the
        # standard memory/gradient trade-off used by Mamba and S4 trainers.
        x_state = x_chunk[:, :, -1, :].detach()

        # Output: y[t] = Σ_n C[t, n] · x[t, n]
        y_chunk = torch.einsum("bdtn,bnt->bdt", x_chunk, C_c)
        ys.append(y_chunk)

    y = torch.cat(ys, dim=2)  # (B, d_in, L)
    y = y + u * D.unsqueeze(0).unsqueeze(-1)
    return y.to(dtype_in)


# ---------------------------------------------------------------------------
# SS2D — 2-D Selective Scan (VMamba-style, 4 directions)
# ---------------------------------------------------------------------------


class SS2D(nn.Module):
    """
    2-D Selective Scan with 4-direction scanning.

    Scans the input feature map in 4 directions and sums the results:
      dir 0: raster   (rows left→right,  top→bottom)
      dir 1: flip-H   (rows left→right,  bottom→top)
      dir 2: flip-W   (rows right→left,  top→bottom)
      dir 3: transpose (columns top→bottom as rows, i.e. (W, H) order)

    Each direction has its own SSM parameters (A, B, C, D, dt) so it can
    learn different temporal dynamics for different spatial contexts.

    I/O format: channel-last  (B, H, W, d_model).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.K = 4
        dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        # ---- input branch ------------------------------------------------
        # Project dim → 2 * d_inner  (SSM input x_  +  gate z)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Short-range context via depthwise conv (applied before SSM)
        self.conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=self.d_inner,
            bias=True,
        )
        self.act = nn.SiLU()

        # ---- SSM parameters for K directions (stored as stacked tensors) -
        # x_proj maps d_inner → (dt_rank + 2 * d_state) per direction
        # NOTE: torch.empty() leaves memory UNINITIALISED — must call init below.
        self.x_proj_weight = nn.Parameter(torch.empty(self.K, self.d_inner, dt_rank + 2 * d_state))
        # dt_proj maps dt_rank → d_inner per direction
        self.dt_proj_weight = nn.Parameter(torch.empty(self.K, dt_rank, self.d_inner))
        self.dt_proj_bias = nn.Parameter(torch.empty(self.K, self.d_inner))
        # A initialised as log of 1…d_state (harmonic sequence → stable SSM)
        A = (
            torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(self.d_inner, -1)
        )  # (d_inner, d_state)
        self.A_log = nn.Parameter(
            torch.log(A).unsqueeze(0).expand(self.K, -1, -1).clone()
        )  # (K, d_inner, d_state)
        self.D = nn.Parameter(torch.ones(self.K, self.d_inner))

        # ---- output branch -----------------------------------------------
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """
        Initialise all raw nn.Parameter tensors that torch.empty() left
        uninitialized.  Missing this step was the primary NaN source:
        x_proj_weight contained arbitrary garbage → NaN B/C/dt from
        the very first forward pass.
        """
        # x_proj: kaiming_uniform per direction (same as nn.Linear default)
        for k in range(self.K):
            nn.init.kaiming_uniform_(self.x_proj_weight[k], a=math.sqrt(5))

        # dt_proj: small uniform so initial dt is in [0.001, 0.1]
        dt_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj_weight, -dt_std, dt_std)
        for k in range(self.K):
            dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            )
            # inv_softplus: softplus(bias) = dt  →  bias = log(expm1(dt))
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                self.dt_proj_bias[k].copy_(inv_dt)

    # ------------------------------------------------------------------
    def _scan_core(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run SS2D on the convolved feature map.

        Fast path  (Kaggle / Linux + CUDA)
        -----------------------------------
        Uses the mamba-ssm CUDA-fused selective_scan_fn.  No Python loop.
        The kernel fuses discretisation + prefix-product scan + output
        projection into a single CUDA kernel with its own memory-efficient
        backward, so per-direction gradient checkpointing is unnecessary.

        At enc1 (B=16, d_inner=192, L=4096):
          Python chunked (old):  64 Python iterations ≈ 13 s / batch
          CUDA kernel    (new):  1 kernel launch      ≈ 0.05 s / batch

        Fallback path  (Windows / CPU)
        --------------------------------
        Pure-Python chunked scan with per-direction gradient checkpointing,
        identical to the original implementation.
        """
        B, C, H, W = x.shape
        L = H * W

        # Build 4 scan sequences — (B, K, d_inner, L)
        xs = torch.stack(
            [
                x.reshape(B, C, L),
                x.flip(2).reshape(B, C, L),
                x.flip(3).reshape(B, C, L),
                x.transpose(2, 3).contiguous().reshape(B, C, L),
            ],
            dim=1,
        )

        # Batched projections
        dbc = torch.einsum("bkdl,kde->bkle", xs, self.x_proj_weight)
        dt = torch.einsum(
            "bkle,ked->bkld",
            dbc[..., : self.dt_rank],
            self.dt_proj_weight,
        ).permute(0, 1, 3, 2)  # (B, K, d_inner, L)
        B_mat = dbc[..., self.dt_rank : self.dt_rank + self.d_state]  # (B, K, L, n)
        C_mat = dbc[..., self.dt_rank + self.d_state :]  # (B, K, L, n)

        A = -torch.exp(self.A_log)  # (K, d_inner, d_state)

        if _MAMBA_CUDA:
            # ---- CUDA-fused path -------------------------------------------
            # selective_scan_fn handles softplus(delta + bias) internally.
            # Returns (B, d_inner, L) per direction — no Python loop needed.
            ys = [
                _mamba_selective_scan(
                    xs[:, k],  # u:          (B, d_inner, L)
                    dt[:, k],  # delta:      (B, d_inner, L)
                    A[k],  # A:          (d_inner, d_state) negative
                    B_mat[:, k].permute(0, 2, 1),  # B:          (B, d_state, L)
                    C_mat[:, k].permute(0, 2, 1),  # C:          (B, d_state, L)
                    self.D[k],  # D:          (d_inner,)
                    delta_bias=self.dt_proj_bias[k],
                    delta_softplus=True,
                )
                for k in range(self.K)
            ]
        else:
            # ---- Python chunked fallback -----------------------------------
            # Per-direction checkpoint stores only I/O tensors, not all 64
            # chunks' intermediate activations simultaneously.
            def _one_dir(xk, dtk, Ak, Bk, Ck, Dk, bias_k):
                return _ssm_scan_chunked(
                    xk, dtk, Ak, Bk.permute(0, 2, 1), Ck.permute(0, 2, 1), Dk, bias_k
                )

            def _scan_dir(k):
                if self.training:
                    return torch.utils.checkpoint.checkpoint(
                        _one_dir,
                        xs[:, k],
                        dt[:, k],
                        A[k],
                        B_mat[:, k],
                        C_mat[:, k],
                        self.D[k],
                        self.dt_proj_bias[k],
                        use_reentrant=False,
                    )
                return _one_dir(
                    xs[:, k],
                    dt[:, k],
                    A[k],
                    B_mat[:, k],
                    C_mat[:, k],
                    self.D[k],
                    self.dt_proj_bias[k],
                )

            ys = [_scan_dir(k) for k in range(self.K)]

        y0 = ys[0].reshape(B, C, H, W)
        y1 = ys[1].reshape(B, C, H, W).flip(2)
        y2 = ys[2].reshape(B, C, H, W).flip(3)
        y3 = ys[3].reshape(B, C, W, H).transpose(2, 3)

        return y0 + y1 + y2 + y3

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): (B, H, W, d_model)  channel-last.

        Returns:
            Tensor: (B, H, W, d_model)  channel-last.
        """
        xz = self.in_proj(x)  # (B, H, W, 2 * d_inner)
        x_, z = xz.chunk(2, dim=-1)  # each (B, H, W, d_inner)

        x_ = x_.permute(0, 3, 1, 2).contiguous()  # → (B, d_inner, H, W)
        x_ = self.act(self.conv2d(x_))  # local short-range context

        y = self._scan_core(x_)  # (B, d_inner, H, W)
        y = y.permute(0, 2, 3, 1)  # → (B, H, W, d_inner)

        y = self.out_norm(y) * F.silu(z)  # output gate
        return self.out_proj(y)  # (B, H, W, d_model)


# ---------------------------------------------------------------------------
# VSSBlock & VSSStage
# ---------------------------------------------------------------------------


class VSSBlock(nn.Module):
    """Residual Visual State Space Block: LayerNorm → SS2D → residual add."""

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(d_model=dim, d_state=d_state, d_conv=d_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, dim)  channel-last."""
        return x + self.ss2d(self.norm(x))


class VSSStage(nn.Module):
    """
    N stacked VSSBlocks operating at a fixed spatial resolution.

    Args:
        dim           (int):  Channel width.
        depth         (int):  Number of VSSBlocks.
        d_state       (int):  SSM state dimension.
        use_checkpoint(bool): If True, use gradient checkpointing to trade
                              compute for memory (recomputes activations
                              during backward instead of storing them).
                              Recommended for heavy stages (enc3, bottleneck).
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int = 16,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([VSSBlock(dim, d_state=d_state) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, dim)  channel-last."""
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x


# ---------------------------------------------------------------------------
# Spatial transition layers (all channel-last I/O)
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """
    Stride-4 embedding using two stride-2 Conv3×3.

    Rationale: first VSSBlocks work at H/4 × W/4 → max sequence length
    4 096 at 256 px, which keeps the chunked scan fast and numerically stable.

    Args:
        in_channels (int): Input image channels (3, 4, or 5).
        dim         (int): Output feature channels (= dims[0]).
    """

    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        mid = max(dim // 2, 16)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, H/4, W/4, dim)  channel-last."""
        return self.net(x).permute(0, 2, 3, 1).contiguous()


class DownLayer(nn.Module):
    """Stride-2 downsampling with channel doubling.  Channel-last I/O."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, in_dim) → (B, H/2, W/2, out_dim)."""
        x = x.permute(0, 3, 1, 2).contiguous()  # → channel-first
        return self.conv(x).permute(0, 2, 3, 1)  # → channel-last


class UpLayer(nn.Module):
    """
    ConvTranspose2d upsample + skip-concatenation + Conv projection.
    Channel-last I/O.
    """

    def __init__(self, in_dim: int, skip_dim: int, out_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_dim, in_dim, kernel_size=2, stride=2)
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim + skip_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    (Tensor): (B, H,  W,  in_dim)   — from deeper decoder level.
            skip (Tensor): (B, H', W', skip_dim) — corresponding encoder skip.

        Returns:
            Tensor: (B, H', W', out_dim)  channel-last.
        """
        x = self.up(x.permute(0, 3, 1, 2))  # (B, in_dim,   2H, 2W)
        skip = skip.permute(0, 3, 1, 2)  # (B, skip_dim, H', W')

        # Pad if spatial dims are odd (H' may differ by ±1 from 2H)
        dH = skip.size(2) - x.size(2)
        dW = skip.size(3) - x.size(3)
        if dH != 0 or dW != 0:
            x = F.pad(x, [dW // 2, dW - dW // 2, dH // 2, dH - dH // 2])

        x = self.proj(torch.cat([skip, x], dim=1))  # (B, out_dim, H', W')
        return x.permute(0, 2, 3, 1)  # (B, H', W', out_dim)


# ---------------------------------------------------------------------------
# MambaUNet
# ---------------------------------------------------------------------------


class MambaUNet(nn.Module):
    """
    Native Mamba U-Net for underwater image restoration.

    All encoder and decoder stages use VSS (Visual State Space) blocks,
    giving Mamba's global long-range context throughout the full hierarchy.

    Args:
        in_channels  (int): Input channels — 3 (RGB), 4, or 5 (physics-guided).
        out_channels (int): Output channels. Default 3.
        model_size   (str): ``"T"`` (Tiny, ~24 M params) | ``"S"`` | ``"B"``.
        d_state      (int): SSM state dimension. Default 16.

    Example::

        model = MambaUNet(in_channels=3).cuda()
        y = model(torch.randn(2, 3, 256, 256).cuda())  # (2, 3, 256, 256)
    """

    _CONFIGS: dict[str, dict] = {
        # dims:       (enc1, enc2, enc3, bn)
        # enc_depths: VSSBlocks at each encoder level
        # dec_depths: VSSBlocks at each decoder level (lighter)
        "T": dict(
            dims=(96, 192, 384, 768),
            enc_depths=(2, 2, 6, 2),
            dec_depths=(2, 2, 2, 2),
        ),
        "S": dict(
            dims=(96, 192, 384, 768),
            enc_depths=(2, 2, 9, 2),
            dec_depths=(2, 2, 2, 2),
        ),
        "B": dict(
            dims=(128, 256, 512, 1024),
            enc_depths=(2, 2, 12, 2),
            dec_depths=(2, 2, 2, 2),
        ),
    }

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_size: str = "T",
        d_state: int = 16,
    ):
        super().__init__()
        if model_size not in self._CONFIGS:
            raise ValueError(
                f"model_size must be one of {list(self._CONFIGS)}; got '{model_size}'."
            )

        cfg = self._CONFIGS[model_size]
        dims = cfg["dims"]
        enc_depths = cfg["enc_depths"]
        dec_depths = cfg["dec_depths"]

        d0, d1, d2, d3 = dims

        # ------------------------------------------------------------------
        # Stem: two stride-2 convs → H/4 × W/4
        # ------------------------------------------------------------------
        self.patch_embed = PatchEmbed(in_channels, d0)

        # ------------------------------------------------------------------
        # Encoder
        # use_checkpoint=True on the heavy stages (enc3, bottleneck) where
        # stored activations are largest. Light stages stay off to keep speed.
        # ------------------------------------------------------------------
        self.enc1 = VSSStage(d0, enc_depths[0], d_state, use_checkpoint=False)
        self.down1 = DownLayer(d0, d1)

        self.enc2 = VSSStage(d1, enc_depths[1], d_state, use_checkpoint=False)
        self.down2 = DownLayer(d1, d2)

        self.enc3 = VSSStage(d2, enc_depths[2], d_state, use_checkpoint=True)
        self.down3 = DownLayer(d2, d3)

        # ------------------------------------------------------------------
        # Bottleneck
        # ------------------------------------------------------------------
        self.bottleneck = VSSStage(d3, enc_depths[3], d_state, use_checkpoint=True)

        # ------------------------------------------------------------------
        # Decoder  (lighter depths; UpLayer handles skip-concat)
        # ------------------------------------------------------------------
        self.up3 = UpLayer(d3, d2, d2)
        self.dec3 = VSSStage(d2, dec_depths[2], d_state, use_checkpoint=True)

        self.up2 = UpLayer(d2, d1, d1)
        self.dec2 = VSSStage(d1, dec_depths[1], d_state, use_checkpoint=False)

        self.up1 = UpLayer(d1, d0, d0)
        self.dec1 = VSSStage(d0, dec_depths[0], d_state, use_checkpoint=False)

        # ------------------------------------------------------------------
        # Head: bilinear ×4 back to full resolution
        # patch_embed did ×4 down; decoder brought it back to H/4.
        # ------------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(d0, d0 // 2, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(d0 // 2, out_channels, kernel_size=1),
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
            Tensor: (N, 3, H, W)  restored image in [0, 1].
        """
        # ---- Stem -------------------------------------------------------
        x = self.patch_embed(x)  # (N, H/4,  W/4,  d0)  channel-last

        # ---- Encoder ----------------------------------------------------
        s1 = self.enc1(x)  # (N, H/4,  W/4,  d0)
        x = self.down1(s1)  # (N, H/8,  W/8,  d1)

        s2 = self.enc2(x)  # (N, H/8,  W/8,  d1)
        x = self.down2(s2)  # (N, H/16, W/16, d2)

        s3 = self.enc3(x)  # (N, H/16, W/16, d2)
        x = self.down3(s3)  # (N, H/32, W/32, d3)

        # ---- Bottleneck -------------------------------------------------
        x = self.bottleneck(x)  # (N, H/32, W/32, d3)

        # ---- Decoder ----------------------------------------------------
        x = self.up3(x, s3)  # (N, H/16, W/16, d2)
        x = self.dec3(x)

        x = self.up2(x, s2)  # (N, H/8,  W/8,  d1)
        x = self.dec2(x)

        x = self.up1(x, s1)  # (N, H/4,  W/4,  d0)
        x = self.dec1(x)

        # ---- Head -------------------------------------------------------
        x = x.permute(0, 3, 1, 2).contiguous()  # → (N, d0, H/4, W/4)
        return self.head(x)  # (N, out_ch, H, W)
