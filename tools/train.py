import os
import sys
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.yolo_sp import YOLOSP_Distiller
from models.losses import OutputDistillationLoss
from data.dataset import DistillDataset


def main():
    # 获取项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'tools', 'checkpoints')

    # 1. 加载配置
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. 构建模型
    model = YOLOSP_Distiller(
        yolo_weights=cfg['weights']['yolo'],
        sp_weights=cfg['weights']['superpoint'],
        device=device
    )

    # 3. 构建数据
    train_img_dir = cfg['data']['train_img_dir']
    val_img_dir = cfg['data']['val_img_dir']

    # 如果配置中的路径是相对路径，转换为绝对路径
    if not os.path.isabs(train_img_dir):
        train_img_dir = os.path.join(PROJECT_ROOT, train_img_dir)
    if not os.path.isabs(val_img_dir):
        val_img_dir = os.path.join(PROJECT_ROOT, val_img_dir)

    train_dataset = DistillDataset(
        img_dir=train_img_dir,
        img_size=cfg['model']['input_size']
    )
    val_dataset = DistillDataset(
        img_dir=val_img_dir,
        img_size=cfg['model']['input_size']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 4. 优化器与损失
    # 只优化 Adapter！
    optimizer = torch.optim.AdamW(
        model.adapter.parameters(),
        lr=float(cfg['train']['distill_lr']),
        weight_decay=1e-4
    )

    # 学习率调度器：余弦退火，防止震荡
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    # 使用新的输出级蒸馏损失
    criterion = OutputDistillationLoss(
        temperature=cfg['train']['temperature'],
        heatmap_weight=1.0,
        desc_weight=0.5
    )

    # 5. 训练循环
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_loss = float('inf')
    epochs = cfg['train']['distill_epochs']
    has_valid_val = len(val_loader) > 0

    print(f"\n=== Starting Distillation Training (Output-Level) ===")
    print(f"Epochs: {epochs}, Batch: {cfg['train']['batch_size']}, LR: {cfg['train']['distill_lr']}")
    print(f"Train images: {len(train_dataset)}, Val images: {len(val_dataset)}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")

    for epoch in range(epochs):
        # ========== Train ==========
        model.train()
        model.yolo.eval()
        model.sp_teacher.eval()

        total_loss = 0.0
        total_heatmap_loss = 0.0
        total_desc_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for imgs in pbar:
            imgs = imgs.to(device)

            # Forward (训练模式，teacher_outputs有值)
            outputs = model(imgs, mode='distill')

            # Loss
            loss_dict = criterion(
                outputs['sp_scores'],
                outputs['sp_descs'],
                outputs['teacher_outputs']
            )
            loss = loss_dict['total']

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_heatmap_loss += loss_dict['heatmap'].item()
            total_desc_loss += loss_dict['descriptor'].item()
            pbar.set_postfix({
                'Loss': f"{loss.item():.6f}",
                'Heat': f"{loss_dict['heatmap'].item():.6f}",
                'Desc': f"{loss_dict['descriptor'].item():.6f}"
            })

        # Scheduler
        scheduler.step()

        # ========== Validation ==========
        avg_train_loss = total_loss / len(train_loader)
        avg_heatmap = total_heatmap_loss / len(train_loader)
        avg_desc = total_desc_loss / len(train_loader)

        if has_valid_val:
            # 切换到train模式以获取teacher_outputs
            model.train()
            model.yolo.eval()
            model.sp_teacher.eval()

            val_loss = 0.0
            val_heatmap = 0.0
            val_desc = 0.0
            val_samples = 0

            with torch.no_grad():
                for imgs in val_loader:
                    imgs = imgs.to(device)
                    outputs = model(imgs, mode='distill')

                    # 只有teacher_outputs有值时才计算损失
                    if outputs['teacher_outputs'] is not None:
                        loss_dict = criterion(
                            outputs['sp_scores'],
                            outputs['sp_descs'],
                            outputs['teacher_outputs']
                        )
                        val_loss += loss_dict['total'].item()
                        val_heatmap += loss_dict['heatmap'].item()
                        val_desc += loss_dict['descriptor'].item()
                        val_samples += 1

            if val_samples > 0:
                avg_val_loss = val_loss / val_samples
                avg_val_heatmap = val_heatmap / val_samples
                avg_val_desc = val_desc / val_samples
            else:
                avg_val_loss = float('inf')
                avg_val_heatmap = 0.0
                avg_val_desc = 0.0

            print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.6f} (Heat={avg_heatmap:.6f}, Desc={avg_desc:.6f}), Val Loss={avg_val_loss:.6f} (Heat={avg_val_heatmap:.6f}, Desc={avg_val_desc:.6f})")

            # 保存最佳模型（只有验证集有效且loss下降时才保存）
            if avg_val_loss < best_loss and avg_val_loss < float('inf'):
                best_loss = avg_val_loss
                save_path = os.path.join(CHECKPOINT_DIR, "best_adapter.pth")
                torch.save({
                    'epoch': epoch,
                    'adapter': model.adapter.state_dict(),
                    'loss': best_loss,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss
                }, save_path)
                print(f"  -> Best model saved to {save_path}")
        else:
            # 没有验证集时，基于训练损失保存
            print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.6f} (Heat={avg_heatmap:.6f}, Desc={avg_desc:.6f}), Val: N/A")

            # 每5个epoch保存一次检查点
            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(CHECKPOINT_DIR, f"adapter_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'adapter': model.adapter.state_dict(),
                    'loss': avg_train_loss,
                    'train_loss': avg_train_loss
                }, save_path)
                print(f"  -> Checkpoint saved to {save_path}")

    print("\n=== Training Complete! ===")
    print(f"Best validation loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
