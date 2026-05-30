import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sam2.build_sam import build_sam2
from timm.layers import trunc_normal_


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )
        self.init_weights()

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.prompt_learn.apply(_init_weights)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2=None):
        if x2 is not None:
            diffY = x1.size()[2] - x2.size()[2]
            diffX = x1.size()[3] - x2.size()[3]
            x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        x = self.up(x)
        return self.conv(x)


class FusionAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, sam_feat, hr_feat):
        concat_feat = torch.cat([sam_feat, hr_feat], dim=1)
        att_weight = self.att(concat_feat)
        fused_feat = sam_feat * att_weight + hr_feat * (1 - att_weight)
        return fused_feat


class SAM2HRNet(nn.Module):
    def __init__(self, checkpoint_path=None, hrnet_path=None, for_inference=False) -> None:
        super(SAM2HRNet, self).__init__()

        # ===== SAM2 Encoder =====
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            self.sam_model = build_sam2(model_cfg, checkpoint_path)
        else:
            self.sam_model = build_sam2(model_cfg)
        self.sam_encoder = self.sam_model.image_encoder
        for p in self.sam_encoder.parameters():
            p.requires_grad = False

        # Adapter wrapping
        if hasattr(self.sam_encoder, 'blocks'):
            blocks_src = list(self.sam_encoder.blocks)
            wrapped = []
            for blk in blocks_src:
                wrapped.append(Adapter(blk))
            self.sam_encoder.blocks = nn.Sequential(*wrapped)
        elif hasattr(self.sam_encoder, 'trunk') and hasattr(self.sam_encoder.trunk, 'blocks'):
            blocks_src = list(self.sam_encoder.trunk.blocks)
            wrapped = []
            for blk in blocks_src:
                wrapped.append(Adapter(blk))
            self.sam_encoder.trunk.blocks = nn.Sequential(*wrapped)
        else:
            print("[Warning] 未检测到 blocks，未插入 Adapter。")

        # ===== HRNet-W64 =====
        pretrained = not for_inference and (hrnet_path is None or hrnet_path == "")
        self.hrnet = timm.create_model(
            'hrnet_w64',
            features_only=True,
            pretrained=pretrained,
            num_classes=0
        )

        if hrnet_path is not None and hrnet_path != "" and not for_inference:
            try:
                hrnet_weights = torch.load(hrnet_path, map_location='cpu')
                hrnet_weights = {k.replace('module.', ''): v for k, v in hrnet_weights.items()}
                self.hrnet.load_state_dict(hrnet_weights, strict=False)
            except Exception as e:
                print(f"[HRNet] 加载权重失败（忽略）：{e}")

        for name, p in self.hrnet.named_parameters():
            p.requires_grad = True

        # ===== 特征对齐层 =====
        self.align1 = nn.Conv2d(64, 144, 1)
        self.align2 = nn.Conv2d(128, 288, 1)
        self.align3 = nn.Conv2d(256, 576, 1)
        self.align4 = nn.Conv2d(512, 1152, 1)

        # ===== 注意力融合模块 =====
        self.fusion1 = FusionAttention(144)
        self.fusion2 = FusionAttention(288)
        self.fusion3 = FusionAttention(576)
        self.fusion4 = FusionAttention(1152)

        # ===== 特征降维 =====
        self.reduce1 = nn.Conv2d(144, 128, 1)
        self.reduce2 = nn.Conv2d(288, 128, 1)
        self.reduce3 = nn.Conv2d(576, 128, 1)
        self.reduce4 = nn.Conv2d(1152, 128, 1)

        # ===== 上采样与输出头 =====
        self.up1 = Up(256, 128)
        self.up2 = Up(256, 128)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 128)
        self.head = nn.Conv2d(128, 1, 1)

        self._print_trainable_stats()

    def forward(self, x):
        input_size = x.shape[-2:]

        if hasattr(self.sam_encoder, 'trunk'):
            sam_feats = self.sam_encoder.trunk(x)
        else:
            sam_feats = self.sam_encoder(x)

        if isinstance(sam_feats, (list, tuple)) and len(sam_feats) >= 4:
            x1_s, x2_s, x3_s, x4_s = sam_feats[0], sam_feats[1], sam_feats[2], sam_feats[3]
        else:
            if isinstance(sam_feats, torch.Tensor):
                x4_s = sam_feats
                x3_s = F.interpolate(x4_s, scale_factor=2, mode='bilinear', align_corners=True)
                x2_s = F.interpolate(x3_s, scale_factor=2, mode='bilinear', align_corners=True)
                x1_s = F.interpolate(x2_s, scale_factor=2, mode='bilinear', align_corners=True)
            else:
                x1_s = sam_feats[0]
                x2_s = sam_feats[1] if len(sam_feats) > 1 else F.interpolate(x1_s, scale_factor=0.5)
                x3_s = sam_feats[2] if len(sam_feats) > 2 else F.interpolate(x2_s, scale_factor=0.5)
                x4_s = sam_feats[-1]

        hr_feats = self.hrnet(x)

        x1_h = self.align1(hr_feats[0])
        x1_h = F.interpolate(x1_h, size=x1_s.shape[-2:], mode='bilinear', align_corners=True)

        x2_h = self.align2(hr_feats[1])
        x2_h = F.interpolate(x2_h, size=x2_s.shape[-2:], mode='bilinear', align_corners=True)

        x3_h = self.align3(hr_feats[2])
        x3_h = F.interpolate(x3_h, size=x3_s.shape[-2:], mode='bilinear', align_corners=True)

        x4_h = self.align4(hr_feats[3])
        x4_h = F.interpolate(x4_h, size=x4_s.shape[-2:], mode='bilinear', align_corners=True)

        x1 = self.fusion1(x1_s, x1_h)
        x2 = self.fusion2(x2_s, x2_h)
        x3 = self.fusion3(x3_s, x3_h)
        x4 = self.fusion4(x4_s, x4_h)

        x1, x2, x3, x4 = self.reduce1(x1), self.reduce2(x2), self.reduce3(x3), self.reduce4(x4)
        x = self.up4(x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        out = self.head(x)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        return out

    def _print_trainable_stats(self):
        total = 0
        trainable = 0
        names_trainable = []
        for name, p in self.named_parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
                names_trainable.append(name)
        print(f"[Model] total params: {total:,} | trainable params: {trainable:,} ({trainable/total:.2%})")
        sample = names_trainable[:40]
        if sample:
            print("[Model] 示例可训练参数名（最多显示40项）:")
            for s in sample:
                print("   ", s)
        else:
            print("[Model] 没有检测到可训练参数，请检查 freeze/unfreeze 逻辑。")