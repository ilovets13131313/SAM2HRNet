import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from contextlib import nullcontext
from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.metrics import jaccard_score

# 你提供的模型文件（已经按 SAM2UNet 策略修改）
from SAM2HRNet1 import SAM2HRNet

# 你提供的 dataset（假设保存在 dataset.py 中）
from dataset import FullDataset

# -----------------------
# 损失与指标
# -----------------------
def structure_loss(pred, mask):
    if pred.shape[-2:] != mask.shape[-2:]:
        pred = F.interpolate(pred, size=mask.shape[-2:], mode='bilinear', align_corners=True)
    mask = mask.float()
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred_sigmoid = torch.sigmoid(pred)
    inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1e-6) / (union - inter + 1e-6)
    return (wbce + wiou).mean()


def calculate_metrics(pred, mask, threshold=0.5, is_logits=True):
    pred_sigmoid = torch.sigmoid(pred) if is_logits else pred
    pred_binary = (pred_sigmoid > threshold).float()
    mask_binary = (mask > threshold).float()

    tp = (pred_binary * mask_binary).sum(dim=(2, 3))
    fp = (pred_binary * (1 - mask_binary)).sum(dim=(2, 3))
    fn = ((1 - pred_binary) * mask_binary).sum(dim=(2, 3))

    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)

    return {
        'iou': iou.mean().item(),
        'dice': dice.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item()
    }

def calculate_iou2(pred_np, mask_np, threshold=0.5):
    pred_bin = (pred_np >= threshold * 255).astype(np.uint8)
    mask_bin = (mask_np >= 0.5 * 255).astype(np.uint8)
    return jaccard_score(mask_bin.flatten(), pred_bin.flatten())

# -----------------------
# Helper: build optimizer param groups
# -----------------------
def build_param_groups(model, base_lr):
    """
    给可训练参数分组：
     - sam 相关 (名字中含 'sam_encoder') 使用 base_lr/10
     - 其他可训练参数使用 base_lr
    只包含 requires_grad == True 的参数
    """
    sam_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'sam_encoder' in name.lower():
            sam_params.append(p)
        else:
            other_params.append(p)
    groups = []
    if len(sam_params) > 0:
        groups.append({'params': sam_params, 'lr': base_lr / 10})
    if len(other_params) > 0:
        groups.append({'params': other_params, 'lr': base_lr})
    # Safety: if nothing matched, fall back to all trainable params
    if len(groups) == 0:
        groups = [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': base_lr}]
    return groups

# -----------------------
# 主训练函数
# -----------------------
def main(args):
    # 设备选择
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device.lower() else 'cpu')
    amp_autocast = (torch.cuda.amp.autocast if device.type == 'cuda' else nullcontext)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # 数据集与 dataloader
    train_dataset = FullDataset(args.train_image_path, args.train_mask_path, args.img_size, mode='train')
    val_dataset = FullDataset(args.val_image_path, args.val_mask_path, args.img_size, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type=='cuda'), drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type=='cuda'), drop_last=False)

    # 模型
    model = SAM2HRNet(args.sam_checkpoint, args.hrnet_checkpoint).to(device)

    # 确保训练脚本只优化 requires_grad=True 的参数
    # 构造参数组（sam adapter lr 较小，其余 base_lr）
    param_groups = build_param_groups(model, args.base_lr)
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # Scheduler: warmup (LinearLR) + CosineAnnealingLR 按 step（iter）
    warmup_steps = int(args.warmup_epochs * len(train_loader))
    total_steps = int((args.epochs - args.warmup_epochs) * len(train_loader))
    if warmup_steps <= 0:
        warmup_steps = 0
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps) if warmup_steps > 0 else None
    main_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=1e-7)
    if warmup_scheduler is not None:
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler],
                                 milestones=[warmup_steps])
    else:
        scheduler = main_scheduler  # fallback

    os.makedirs(args.save_path, exist_ok=True)
    best_dice = 0.0

    print("[Info] Start training")
    # 打印模型可训练参数信息（示例）
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] total params: {total_p:,}, trainable: {trainable_p:,} ({trainable_p/total_p:.2%})")
    # show some trainable parameter names
    shown = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(" trainable:", name)
            shown += 1
            if shown >= 40:
                break

    # 训练主循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Train")
        for step, batch in enumerate(pbar_train):
            imgs = batch['image'].to(device)       # [B,3,H,W]
            masks = batch['label'].to(device)      # [B,1,H,W] or [B,H,W] but dataset returns 1-channel

            # forward + loss (AMP if available)
            with amp_autocast():
                preds = model(imgs)
                # 如果模型返回 tuple/list（兼容SAM2UNet式输出），取第一个作为主输出
                if isinstance(preds, (tuple, list)):
                    preds = preds[0]
                # 保证 preds shape 为 [B,1,H,W]
                if preds.dim() == 4 and preds.shape[1] != 1:
                    # 如果模型返回多通道概率 (C>1)，取第1通道或平均（这里取第0通道）
                    preds = preds[:, 0:1, ...]
                loss = structure_loss(preds, masks)
                loss = loss / args.accumulation_steps

            # backward
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # accumulation step
            if (step + 1) % args.accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                # scheduler step after optimizer.step()
                try:
                    scheduler.step()
                except Exception:
                    # if scheduler is SequentialLR, step() ok; else if it's main_scheduler it's ok
                    pass
                optimizer.zero_grad()

            # logging
            bs = imgs.size(0)
            total_loss += loss.item() * args.accumulation_steps * bs  # 恢复为未除以 accumulation 的 loss
            total_samples += bs
            pbar_train.set_postfix(loss=total_loss / max(1, total_samples))

        avg_train_loss = total_loss / max(1, total_samples)
        print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {avg_train_loss:.6f}")

        # 每 10 轮保存一次模型
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.save_path, f"SAM2HRNet-{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
            print(f"[Info] Saved model: {save_path}")

        # 每 50 轮保存一次测试掩码
        if (epoch + 1) % 50 == 0:
            test_mask_dir = os.path.join(args.save_path, f"test_masks_epoch_{epoch+1}")
            os.makedirs(test_mask_dir, exist_ok=True)
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} Save Test Masks")):
                    imgs = batch['image'].to(device)
                    preds = model(imgs)
                    if isinstance(preds, (tuple, list)):
                        preds = preds[0]
                    if preds.dim() == 4 and preds.shape[1] != 1:
                        preds = preds[:, 0:1, ...]
                    pred_sigmoid = torch.sigmoid(preds)
                    pred_binary = (pred_sigmoid > 0.5).float().cpu().numpy() * 255
                    for j in range(pred_binary.shape[0]):
                        img_name = f"val_{i * args.batch_size + j}.png"
                        Image.fromarray(pred_binary[j, 0].astype(np.uint8)).save(os.path.join(test_mask_dir, img_name))

        # Validation every val_interval epochs
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            total_val_loss = 0.0
            total_iou1 = 0.0
            total_iou2 = 0.0
            total_dice = 0.0
            total_precision = 0.0
            total_recall = 0.0
            val_samples = 0

            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} Val")
            with torch.no_grad():
                for batch in pbar_val:
                    imgs = batch['image'].to(device)
                    masks = batch['label'].to(device)
                    bs = imgs.size(0)
                    preds = model(imgs)
                    if isinstance(preds, (tuple, list)):
                        preds = preds[0]
                    if preds.dim() == 4 and preds.shape[1] != 1:
                        preds = preds[:, 0:1, ...]
                    loss = structure_loss(preds, masks)
                    total_val_loss += loss.item() * bs

                    metrics = calculate_metrics(preds, masks, is_logits=True)
                    total_iou1 += metrics['iou'] * bs
                    total_dice += metrics['dice'] * bs
                    total_precision += metrics['precision'] * bs
                    total_recall += metrics['recall'] * bs

                    # 计算 IoU2
                    pred_np = (torch.sigmoid(preds) > 0.5).float().cpu().numpy() * 255
                    mask_np = masks.cpu().numpy() * 255
                    for j in range(bs):
                        total_iou2 += calculate_iou2(pred_np[j, 0], mask_np[j, 0])

                    val_samples += bs
                    pbar_val.set_postfix(
                        loss=total_val_loss / max(1, val_samples),
                        IoU1=total_iou1 / max(1, val_samples),
                        IoU2=total_iou2 / max(1, val_samples),
                        Dice=total_dice / max(1, val_samples)
                    )

            avg_val_loss = total_val_loss / max(1, val_samples)
            avg_iou1 = total_iou1 / max(1, val_samples)
            avg_iou2 = total_iou2 / max(1, val_samples)
            avg_dice = total_dice / max(1, val_samples)
            avg_precision = total_precision / max(1, val_samples)
            avg_recall = total_recall / max(1, val_samples)

            print(f"[Val] Epoch [{epoch+1}/{args.epochs}] Loss: {avg_val_loss:.6f} | IoU1: {avg_iou1:.4f} | IoU2: {avg_iou2:.4f} | Dice: {avg_dice:.4f} | Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f}")

            # 保存最佳 dice
            if avg_dice > best_dice:
                best_dice = avg_dice
                save_path = os.path.join(args.save_path, "best_dice.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice
                }, save_path)
                print(f"[Info] Saved best dice model: {best_dice:.4f} -> {save_path}")

    print("Training finished.")


# -----------------------
# CLI 参数
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--hrnet_checkpoint", type=str, required=True)
    parser.add_argument("--train_image_path", type=str, required=True)
    parser.add_argument("--train_mask_path", type=str, required=True)
    parser.add_argument("--val_image_path", type=str, required=True)
    parser.add_argument("--val_mask_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=2e-5)
    parser.add_argument("--img_size", type=int, default=352)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--val_interval", type=int, default=10,
                        help="validation every N epochs")
    args = parser.parse_args()

    main(args)