import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# 1. Bloques y U-Net 1D (tu código original)
# =========================

class ConvBlock1D(nn.Module):
    """
    Bloque conv -> (BN) -> ReLU -> conv -> (BN) -> ReLU
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = kernel_size // 2

        layers = []
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


def center_crop_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Recorta el eje temporal al centro para que coincida con target_len.
    x: (B, C, T)
    """
    _, _, T = x.shape
    if T == target_len:
        return x
    if T < target_len:
        pad = target_len - T
        return F.pad(x, (0, pad))
    start = (T - target_len) // 2
    end = start + target_len
    return x[..., start:end]


class UNet1D(nn.Module):
    """
    U-Net 1D configurable, para denoising:
      - Entrada: (B, in_channels, T)
      - Salida:  (B, out_channels, T)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        num_layers: int = 4,
        kernel_size: int = 15,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        final_activation: str | None = None,  # None, "tanh", "sigmoid"
    ):
        super().__init__()

        self.final_activation = final_activation

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        enc_channels = []

        current_in = in_channels
        for i in range(num_layers):
            out_ch = base_channels * (2 ** i)
            self.enc_blocks.append(
                ConvBlock1D(
                    in_channels=current_in,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                )
            )
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            enc_channels.append(out_ch)
            current_in = out_ch

        # Bottleneck
        bottleneck_channels = base_channels * (2 ** num_layers)
        self.bottleneck = ConvBlock1D(
            in_channels=current_in,
            out_channels=bottleneck_channels,
            kernel_size=kernel_size,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )

        # Decoder
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        current_in = bottleneck_channels
        for i in reversed(range(num_layers)):
            skip_ch = enc_channels[i]

            self.upconvs.append(
                nn.ConvTranspose1d(
                    in_channels=current_in,
                    out_channels=skip_ch,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.dec_blocks.append(
                ConvBlock1D(
                    in_channels=skip_ch * 2,
                    out_channels=skip_ch,
                    kernel_size=kernel_size,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                )
            )
            current_in = skip_ch

        # Capa final
        self.final_conv = nn.Conv1d(current_in, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, T)
        enc_feats = []

        # Encoder
        out = x
        for block, pool in zip(self.enc_blocks, self.pools):
            out = block(out)
            enc_feats.append(out)
            out = pool(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder
        for upconv, dec_block, skip in zip(self.upconvs, self.dec_blocks, reversed(enc_feats)):
            out = upconv(out)
            if out.shape[-1] != skip.shape[-1]:
                out = center_crop_1d(out, skip.shape[-1])
            out = torch.cat([skip, out], dim=1)
            out = dec_block(out)

        out = self.final_conv(out)

        if self.final_activation == "tanh":
            out = torch.tanh(out)
        elif self.final_activation == "sigmoid":
            out = torch.sigmoid(out)

        return out

# =========================
# 2. Bloques y U-Net 2D (nuevo)
# =========================

class ConvBlock2D(nn.Module):
    """
    Bloque conv2d -> (BN) -> ReLU -> conv2d -> (BN) -> ReLU
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = kernel_size // 2

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


def center_crop_2d(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Recorta (H, W) al centro para que coincida con (target_h, target_w).
    x: (B, C, H, W)
    """
    _, _, H, W = x.shape
    # Eje H
    if H < target_h:
        pad_h = target_h - H
        x = F.pad(x, (0, 0, 0, pad_h))  # (left, right, top, bottom) sobre (W, H)
        H = target_h
    if W < target_w:
        pad_w = target_w - W
        x = F.pad(x, (0, pad_w, 0, 0))
        W = target_w

    if H > target_h:
        start_h = (H - target_h) // 2
        end_h = start_h + target_h
    else:
        start_h, end_h = 0, H

    if W > target_w:
        start_w = (W - target_w) // 2
        end_w = start_w + target_w
    else:
        start_w, end_w = 0, W

    return x[:, :, start_h:end_h, start_w:end_w]


class UNet2D(nn.Module):
    """
    U-Net 2D para espectrogramas:
      - Entrada: (B, in_channels, F, T)
      - Salida:  (B, out_channels, F, T)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        num_layers: int = 4,
        kernel_size: int = 3,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        final_activation: str | None = None,  # None, "tanh", "sigmoid"
    ):
        super().__init__()

        self.final_activation = final_activation

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        enc_channels = []

        current_in = in_channels
        for i in range(num_layers):
            out_ch = base_channels * (2 ** i)
            self.enc_blocks.append(
                ConvBlock2D(
                    in_channels=current_in,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                )
            )
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            enc_channels.append(out_ch)
            current_in = out_ch

        # Bottleneck
        bottleneck_channels = base_channels * (2 ** num_layers)
        self.bottleneck = ConvBlock2D(
            in_channels=current_in,
            out_channels=bottleneck_channels,
            kernel_size=kernel_size,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )

        # Decoder
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        current_in = bottleneck_channels
        for i in reversed(range(num_layers)):
            skip_ch = enc_channels[i]

            self.upconvs.append(
                nn.ConvTranspose2d(
                    in_channels=current_in,
                    out_channels=skip_ch,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.dec_blocks.append(
                ConvBlock2D(
                    in_channels=skip_ch * 2,
                    out_channels=skip_ch,
                    kernel_size=kernel_size,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                )
            )
            current_in = skip_ch

        # Capa final
        self.final_conv = nn.Conv2d(current_in, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, F, T)
        enc_feats = []

        # Encoder
        out = x
        for block, pool in zip(self.enc_blocks, self.pools):
            out = block(out)
            enc_feats.append(out)
            out = pool(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder
        for upconv, dec_block, skip in zip(self.upconvs, self.dec_blocks, reversed(enc_feats)):
            out = upconv(out)
            _, _, H_skip, W_skip = skip.shape
            _, _, H_out, W_out = out.shape
            if (H_out != H_skip) or (W_out != W_skip):
                out = center_crop_2d(out, H_skip, W_skip)

            out = torch.cat([skip, out], dim=1)  # (B, C_skip + C_up, F, T)
            out = dec_block(out)

        out = self.final_conv(out)

        if self.final_activation == "tanh":
            out = torch.tanh(out)
        elif self.final_activation == "sigmoid":
            out = torch.sigmoid(out)

        return out
