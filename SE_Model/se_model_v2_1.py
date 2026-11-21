import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer import ConformerBlock


def power_compress(x):
    mag = torch.abs(x)
    phase = torch.angle(x)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1)


def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** (1.0 / 0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)


def pad_to_multiple(x, mode="dim_2", multiple=16):
    pad_f = 0
    pad_t = 0

    _, _, T, Freq = x.shape

    if mode == "dim_2":
        pad_t = (multiple - T % multiple) % multiple
    elif mode == "dim_3":
        pad_f = (multiple - Freq % multiple) % multiple
    else:
        pad_f = (multiple - Freq % multiple) % multiple
        pad_t = (multiple - T % multiple) % multiple

    return F.pad(x, (0, pad_f, 0, pad_t), mode="constant", value=0.0)
    
    
    
class HarmonicAttention(nn.Module):
    def __init__(self, in_channels, out_channels, is_final=False):
        super(HarmonicAttention, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # effective initial alpha = 0.65 * tanh(0.077) ˜ 0.05
        self.channel_alpha = nn.Parameter(torch.ones(out_channels) * 0.077)

        # mask-side projections
        self.mask_proj = nn.Conv2d(1, out_channels, kernel_size=1, bias=True)
        self.comp_proj = nn.Conv2d(1, out_channels, kernel_size=1, bias=True)
        self.frame_proj = nn.Conv2d(1, out_channels, kernel_size=1, bias=True)

        # important: revert to v2 setting
        # sigmoid(0) = 0.5, so complement path is available but not overly strong
        self.uv_lambda = nn.Parameter(torch.zeros(out_channels))

        # keep v2 smoothing strength
        self.smooth_alpha = nn.Parameter(torch.tensor(-1.5))  # sigmoid(-1.5) ˜ 0.18

        # guarded frame-center parameter
        # sigmoid(-0.2007) ˜ 0.45
        self.frame_center_logit = nn.Parameter(torch.tensor(-0.85))
        self.frame_scale = nn.Parameter(torch.tensor(0.0))

        # boundary/detail path, but controlled
        self.boundary_proj = nn.Conv2d(1, out_channels, kernel_size=1, bias=True)
        self.boundary_alpha = nn.Parameter(torch.ones(out_channels) * -2.0)

        # learnable boundary sharpness
        self.boundary_sharpness = nn.Parameter(torch.tensor(0.5))

        if not is_final:
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
            self.activation = nn.PReLU(out_channels)

        self.is_final = is_final

        # initialization for sigmoid-bounded mask inputs
        nn.init.xavier_uniform_(self.mask_proj.weight)
        nn.init.zeros_(self.mask_proj.bias)

        nn.init.xavier_uniform_(self.comp_proj.weight)
        nn.init.zeros_(self.comp_proj.bias)

        # frame branch starts neutral
        nn.init.zeros_(self.frame_proj.weight)
        nn.init.zeros_(self.frame_proj.bias)

        # boundary branch starts as no-op
        nn.init.zeros_(self.boundary_proj.weight)
        nn.init.zeros_(self.boundary_proj.bias)

    def forward(self, x, harmonic_mask):

        x = self.conv(x)

        if not self.is_final:
            x = self.norm(x)

        # safe soft mask conversion
        harmonic_mask = torch.sigmoid(harmonic_mask)

        # mild learnable frequency smoothing
        smooth_alpha = torch.sigmoid(self.smooth_alpha)
        harmonic_mask_s = (1.0 - smooth_alpha) * harmonic_mask + smooth_alpha * F.avg_pool2d(
            harmonic_mask,
            kernel_size=(1, 5),
            stride=1,
            padding=(0, 2)
        )

        # frame-level harmonic confidence
        frame_conf = harmonic_mask_s.mean(dim=-1, keepdim=True)  # [B,1,T,1]
        frame_conf = F.avg_pool2d(
            frame_conf,
            kernel_size=(5, 1),
            stride=1,
            padding=(2, 0)
        )

        # guarded voiced/unvoiced calibration
        frame_center = torch.sigmoid(self.frame_center_logit)
        frame_scale = 1.0 + F.softplus(self.frame_scale)
        frame_mix = torch.sigmoid(frame_scale * (frame_conf - frame_center))  # [B,1,T,1]

        # boundary/detail confidence from mask transition regions
        boundary_conf = torch.abs(
            harmonic_mask_s - F.avg_pool2d(
                harmonic_mask_s,
                kernel_size=(1, 7),
                stride=1,
                padding=(0, 3)
            )
        )

        boundary_sharpness = 1.0 + F.softplus(self.boundary_sharpness)
        boundary_conf = torch.tanh(boundary_sharpness * boundary_conf)

        # important fix:
        # boundary path only active in frames with voiced/harmonic confidence
        boundary_conf = boundary_conf * frame_mix

        # logits from harmonic, complement, frame, and boundary paths
        harm_logits = self.mask_proj(harmonic_mask_s)            # [B,C,T,F]
        comp_logits = self.comp_proj(1.0 - harmonic_mask_s)      # [B,C,T,F]
        frame_logits = self.frame_proj(frame_mix)                # [B,C,T,1]
        boundary_logits = self.boundary_proj(boundary_conf)      # [B,C,T,F]

        # per-channel complement usage
        uv_mix = torch.sigmoid(self.uv_lambda).view(1, -1, 1, 1)

        # voiced frames -> harmonic dominates
        # weak/unvoiced frames -> complement is allowed, but not over-opened
        harm_mix = frame_mix + (1.0 - frame_mix) * (1.0 - uv_mix)
        comp_mix = (1.0 - frame_mix) * uv_mix

        # small learned boundary correction
        boundary_mix = torch.sigmoid(self.boundary_alpha).view(1, -1, 1, 1)

        gate_logits = (
            harm_mix * harm_logits
            + comp_mix * comp_logits
            + frame_logits
            + boundary_mix * boundary_logits
        )

        gate = torch.sigmoid(gate_logits)

        # bounded modulation strength
        alpha = 0.65 * torch.tanh(self.channel_alpha).view(1, -1, 1, 1)

        x = x + alpha * (x * gate)

        if not self.is_final:
            x = self.activation(x)

        return x


def icnr_(weight, upscale=2):
    """
    ICNR init (Aitken et al. 2017) for the sub-pixel projection.

    The polyphase filters start identical, so the split behaves as
    nearest-neighbour interpolation at step 0 and cannot produce a checkerboard
    before it has learned anything. Channel layout must be c * upscale + j,
    which is what FrequencySplit unpacks.
    """
    out_ch = weight.shape[0]
    assert out_ch % upscale == 0, "out_channels must be divisible by upscale"

    sub = torch.empty(out_ch // upscale, *weight.shape[1:],
                      device=weight.device, dtype=weight.dtype)
    nn.init.xavier_uniform_(sub)

    with torch.no_grad():
        weight.copy_(sub.repeat_interleave(upscale, dim=0))


class FrequencyMerge(nn.Module):
    """
    F -> ceil(F/2), information-preserving.

        neighbouring bin pair -> channels -> 1x1 Conv

    Bin 2k goes to channel group 0 and bin 2k+1 to channel group 1, so the
    comb structure survives the reduction intact and the 1x1 decides what to
    keep. Contrast with blur-pool, which low-passes the comb away before
    throwing every second bin out.

    Odd F is padded by one bin; FrequencySplit crops it back.
    """

    def __init__(self, in_channels, out_channels, f_in):
        super().__init__()
        self.pad_f = f_in % 2

        self.proj = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = nn.PReLU(out_channels)

    def forward(self, x):                                   # [B, C, T, F]
        if self.pad_f:
            x = F.pad(x, (0, self.pad_f))

        B, C, T, Fb = x.shape
        x = x.reshape(B, C, T, Fb // 2, 2)
        x = x.permute(0, 1, 4, 2, 3).contiguous()           # [B, C, 2, T, F/2]
        x = x.reshape(B, C * 2, T, Fb // 2)                 # ch = c * 2 + j

        return self.act(self.norm(self.proj(x)))


class FrequencySplit(nn.Module):
    """
    f_in -> f_out (~2x), the exact dual of FrequencyMerge.

        1x1 Conv (C -> 2C) -> interleave channel pairs back into bins

    mode="shuffle" (default) is lossless. mode="interp" is the bilinear
    fallback: usable, but it is a low-pass, so it reintroduces on the synthesis
    side exactly the comb attenuation the merge removed on the analysis side
    (~0.78 retention at F0 = 100 Hz). Kept only so the two can be compared.
    """

    def __init__(self, channels, f_in, f_out, mode="shuffle"):
        super().__init__()
        assert mode in ("shuffle", "interp"), f"unknown resample mode: {mode!r}"
        assert f_out <= 2 * f_in, f"cannot reach F={f_out} from F={f_in}"

        self.mode = mode
        self.f_out = f_out

        if mode == "shuffle":
            self.proj = nn.Conv2d(channels, channels * 2, kernel_size=1)
            icnr_(self.proj.weight, upscale=2)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):                                   # [B, C, T, f_in]
        if self.mode == "interp":
            return F.interpolate(x, size=(x.shape[2], self.f_out),
                                 mode="bilinear", align_corners=True)

        x = self.proj(x)                                    # [B, 2C, T, f_in]
        B, C2, T, Fh = x.shape
        C = C2 // 2

        x = x.reshape(B, C, 2, T, Fh)                       # ch = c * 2 + j
        x = x.permute(0, 1, 3, 4, 2).contiguous()           # [B, C, T, f_in, 2]
        x = x.reshape(B, C, T, Fh * 2)

        return x[..., :self.f_out]



class CrossBand(nn.Module):
    """
    LN -> F-GConv1d (local, leakage) -> compress C -> F-Linear (global, comb)
       -> expand C -> F-GConv1d (dilated, partial-to-partial) -> LayerScale

    gc1 dilation 1  : STFT mainlobe / adjacent-bin leakage
    gc2 dilation d  : reaches the neighbouring partial (d * df ~ F0).
                      NOTE at 15.625 Hz/bin, d=3 reaches only 47 Hz, below the
                      speech F0 range. For Thai F0 80-350 Hz use d ~ 5-22;
                      d = 10 (156 Hz) is a sensible centre.
    F-Linear        : arbitrary-distance comb relations. Identity-init, and the
                      dominant parameter cost (Cr * F * F ~ 2.1M at C=64, F=514).
    """

    def __init__(self, C, F_bins, r=8, G=8, dilation=10, mask_cond=True):
        super().__init__()
        Cr = max(C // r, 4)

        self.ln = nn.LayerNorm(C)                      # over C only, never over F
        self.gc1 = nn.Conv1d(C, C, 3, padding=1, groups=G)
        self.act1 = nn.PReLU(C)

        self.down = nn.Linear(C, Cr)
        
        self.flin = nn.Parameter(
            torch.eye(F_bins).unsqueeze(0).repeat(Cr, 1, 1)   # [Cr, F, F]
        )
        
        
        self.up = nn.Linear(Cr, C)

        self.gc2 = nn.Conv1d(C, C, 3, padding=dilation,
                             dilation=dilation, groups=G)
        self.act2 = nn.PReLU(C)

        self.gamma = nn.Parameter(torch.zeros(C))      # LayerScale: starts as no-op

        self.mask_cond = mask_cond
        if mask_cond:
            self.mask_embed = nn.Conv1d(2, C, kernel_size=1)
            nn.init.zeros_(self.mask_embed.weight)
            nn.init.zeros_(self.mask_embed.bias)

    def forward(self, x, harmonic_mask=None):          # x: [B, C, T, F]
        B, C, T, Fb = x.shape

        h = x.permute(0, 2, 1, 3).reshape(B * T, C, Fb)          # [BT, C, F]
        h = self.ln(h.transpose(1, 2)).transpose(1, 2)           # pre-norm over C

        if self.mask_cond and harmonic_mask is not None:
            m = torch.sigmoid(harmonic_mask)                     # [B,1,T,F]
            m = m.permute(0, 2, 1, 3).reshape(B * T, 1, Fb)
            h = h + self.mask_embed(torch.cat([m, 1.0 - m], dim=1))

        h = self.act1(self.gc1(h))                               # local
        h = self.down(h.transpose(1, 2))                         # [BT, F, Cr]
        h = torch.einsum('bfc,cgf->bgc', h, self.flin)           # global over F
        h = self.up(h).transpose(1, 2)                           # [BT, C, F]
        h = self.act2(self.gc2(h))                               # dilated local

        out = h.reshape(B, T, C, Fb).permute(0, 2, 1, 3)
        return x + self.gamma.view(1, C, 1, 1) * out


class Encoder(nn.Module):
    """
    HA(mask) -> CrossBand(mask) -> HA(mask) -> FrequencyMerge

    All comb-sensitive processing runs at full frequency resolution and the
    harmonic mask is never resampled, so it stays calibrated.
    down_factor=1 keeps full resolution into the Conformer (2x its cost).
    """

    def __init__(self, in_channels, out_channels, F_bins, r=8, G=8, dilation=10,
                 down_factor=2, mask_cond=True):
        super().__init__()
        assert down_factor in (1, 2), "down_factor must be 1 or 2"

        self.ha1 = HarmonicAttention(in_channels, out_channels)
        self.cross_band = CrossBand(out_channels, F_bins, r=r, G=G,
                                    dilation=dilation, mask_cond=mask_cond)
        self.ha2 = HarmonicAttention(out_channels, out_channels)

        self.merge = (FrequencyMerge(out_channels, out_channels, F_bins)
                      if down_factor == 2 else None)

    def forward(self, x, harmonic_mask):
        x = self.ha1(x, harmonic_mask)
        x = self.cross_band(x, harmonic_mask)
        x = self.ha2(x, harmonic_mask)

        skip = x                       # full-resolution feature for the decoder

        if self.merge is not None:
            x = self.merge(x)

        return x, skip


class GatedSkipFusion(nn.Module):
    """
    Content-adaptive per-(t,f) gate on the encoder skip. Zero-init weights with
    bias -2.0, so the gate is a constant ~0.12 at step 0 and opens only where
    the skip earns it.
    """

    def __init__(self, channels, init_bias=-2.0):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate = nn.Conv2d(2 * channels, channels, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, init_bias)

    def forward(self, x, skip):
        g = torch.sigmoid(self.gate(torch.cat([x, skip], dim=1)))
        return x + g * self.proj(skip)


class Decoder(nn.Module):

    def __init__(self, channels, out_channels, F_in, F_out, resample="shuffle"):
        super().__init__()

        self.split = (FrequencySplit(channels, F_in, F_out, mode=resample)
                      if F_in != F_out else None)
        
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 5), padding=(0, 2), groups=channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size=(1, 5), padding=(0, 8), dilation=(1, 4), groups=channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, out_channels, kernel_size=1),
        )

        nn.init.xavier_uniform_(self.head[-1].weight, gain=0.1)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):                       # [B, C, T, F_in]

        if self.split is not None:
            x = self.split(x)                              # [B, C, T, F_out]

        return self.head(x)                                # [B, 2, T, F_out]



class SE_Model(nn.Module):
    """
    HA(mask) -> CrossBand(mask) -> HA(mask) -> FrequencyMerge
             -> Time Conformer
             -> FrequencySplit -> local Conv -> RI head

    Args:
        in_channels   : 2 (compressed real/imag)
        channels      : feature width C
        F_bins        : frequency bins of the INPUT (must match at runtime)
        r, G          : CrossBand channel-reduction and conv groups
        dilation_enc  : encoder CrossBand gc2 dilation (~10 at 15.625 Hz/bin)
        dilation_dec  : decoder CrossBand gc2 dilation (only used if refine)
        down_factor   : 1 (full resolution) or 2 (frequency merge)
        mask_cond     : condition encoder CrossBand on the harmonic mask
        use_skip      : gated full-resolution skip into the decoder
        resample      : "shuffle" (lossless) or "interp" (bilinear fallback)
        refine        : decoder CrossBand -- OFF by default, costs ~2.1M params
    """

    def __init__(self, in_channels, channels, F_bins, r, G, dilation_enc, dilation_dec,
                 down_factor=2, mask_cond=True, resample="shuffle",
                 conf_num_layers=4, conv_kernel_size=17, heads=8,
                 ff_mult=2, expansion_factor=2, attn_dropout=0.0, ff_dropout=0.0,
                 conv_dropout=0.0):

        super(SE_Model, self).__init__()

        self.F_bins = F_bins
        self.conf_num_layers = conf_num_layers

        self.encoder = Encoder(in_channels=in_channels, out_channels=channels,
                               F_bins=F_bins, r=r, G=G, dilation=dilation_enc,
                               down_factor=down_factor, mask_cond=mask_cond)

        self.con_blocks = nn.ModuleList([
            copy.deepcopy(
                ConformerBlock(
                    dim=channels,
                    dim_head=channels,
                    heads=heads,
                    ff_mult=ff_mult,
                    conv_expansion_factor=expansion_factor,
                    conv_kernel_size=conv_kernel_size,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    conv_dropout=conv_dropout,
                )
            )
            for _ in range(conf_num_layers)
        ])

        F_mid = math.ceil(F_bins / down_factor)
        self.decoder = Decoder(channels=channels, out_channels=in_channels,
                               F_in=F_mid, F_out=F_bins, resample=resample)

    def forward(self, x, h_mask):
        """
        Args:
                x:      [B, C, T, F] compressed noisy spectrogram
                h_mask: [B, 1, T, F] harmonic mask (never resampled)
        Return:
                [B, 2, T, F]
        """
        assert x.shape[-1] == self.F_bins, (
            f"input F={x.shape[-1]} does not match F_bins={self.F_bins} "
            f"(CrossBand's F-Linear is a fixed-size matrix)"
        )

        enc_out, skip = self.encoder(x, h_mask)

        # time attention, one sequence per frequency bin
        b, c, t, f = enc_out.size()
        x_t = enc_out.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)

        for i in range(self.conf_num_layers):
            x_t = self.con_blocks[i](x_t) + x_t

        x = x_t.view(b, f, t, c).permute(0, 3, 2, 1)       # [B, C, T, F/2]

        return self.decoder(x)