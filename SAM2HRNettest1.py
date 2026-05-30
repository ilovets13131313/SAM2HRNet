import os
import argparse  # 新增：导入命令行参数解析库
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# ===== 完整的 SAM2HRNet 模型定义（完全保留你原代码，无任何修改）=====
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


# ===== 新增：命令行参数解析函数（保留你原代码的默认值）=====
def get_args():
    parser = argparse.ArgumentParser(description='SAM2HRNet 推理脚本（命令行版）')
    # 核心路径参数（默认值为你原代码中的路径）
    parser.add_argument('--model-path', '-m', type=str,
                        default=r"F:\MDPI\xunlian\modeltrain\finaltrain\finalsave\SAM2HRNet\模型保存\SAM2HRNet-200.pth",
                        help='SAM2HRNet模型权重文件路径 (.pth)')
    parser.add_argument('--test-img-dir', '-i', type=str,
                        default=r"F:\MDPI\高潮\yuexichutu\image",
                        help='测试图片文件夹路径')
    parser.add_argument('--output-mask-dir', '-o', type=str,
                        default=r"F:\MDPI\高潮\yuexichutu\SAM2HRNet",
                        help='预测掩码输出文件夹路径')
    # 可选配置参数（保留你原代码的默认值）
    parser.add_argument('--img-size', '-s', type=int, default=352,
                        help='图片预处理的resize尺寸（默认352）')
    parser.add_argument('--threshold', '-t', type=float, default=0.3,
                        help='Sigmoid二值化阈值（默认0.3）')
    return parser.parse_args()


# ===== 推理逻辑（仅将硬编码参数替换为命令行参数，核心逻辑完全不变）=====
if __name__ == '__main__':
    # 获取命令行参数
    args = get_args()

    # ===== 测试脚本配置（替换为命令行参数，默认值与你原代码一致）=====
    model_path = args.model_path
    test_img_dir = args.test_img_dir
    output_mask_dir = args.output_mask_dir
    img_size = args.img_size
    threshold = args.threshold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 创建模型并加载权重（完全保留你原代码逻辑）=====
    model = SAM2HRNet(for_inference=True).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # ===== 数据预处理（仅将352改为img_size，其余完全不变）=====
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # 替换硬编码的352为命令行参数
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ===== 创建输出文件夹（完全不变）=====
    os.makedirs(output_mask_dir, exist_ok=True)

    # ===== 推理并保存掩码（仅将0.3改为threshold，其余完全不变）=====
    for img_name in os.listdir(test_img_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_img_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(img_tensor)
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]
                pred_mask = (torch.sigmoid(pred) > threshold).float().cpu().numpy()  # 替换硬编码的0.3

            mask_img = Image.fromarray((pred_mask[0, 0] * 255).astype(np.uint8))
            mask_img.save(os.path.join(output_mask_dir, img_name))
            print(f"✅ {img_name} -> 预测掩码已保存")

    print("🎉 测试完成！")