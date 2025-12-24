"""
重新训练模型 - 基于官方配置 + 优化
训练集: BAU Botanical Garden, BAU Museum, Shapla Bil 1-4, Zinda Park 1-2
验证集: Zinda Park 3
"""

import os
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader
import glob
import random

# 设定随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ==================== 配置参数 ====================
BATCH_SIZE = 6  # 降低batch size以适应GPU内存，2个GPU每个3张
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 2e-5  # 参考官方
NUM_EPOCHS = 150
IMG_SIZE = 1024
EARLY_STOP_PATIENCE = 30
DEVICE_IDS = [0, 2]  # 只使用GPU 0和2，避开GPU 1（被占用）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("🌿 AqUavplant 模型训练 v2")
print("=" * 70)
print(f"训练集: BAU Botanical Garden, BAU Museum,")
print(f"        Shapla Bil 1-4, Zinda Park 1-2")
print(f"验证集: Zinda Park 3")
print(f"Epochs: {NUM_EPOCHS} (Early Stop: {EARLY_STOP_PATIENCE})")
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print("=" * 70)

# ==================== 模型定义 ====================

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, dropout=0.1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64, dropout=0.1)
        self.Conv2 = conv_block(ch_in=64, ch_out=128, dropout=0.1)
        self.Conv3 = conv_block(ch_in=128, ch_out=256, dropout=0.2)
        self.Conv4 = conv_block(ch_in=256, ch_out=512, dropout=0.2)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024, dropout=0.3)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512, dropout=0.2)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256, dropout=0.2)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128, dropout=0.1)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64, dropout=0.1)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        return d1

# ==================== 数据增强 ====================

class SimpleAugmentation:
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, mask):
        # 水平翻转
        if random.random() < self.prob:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        # 垂直翻转
        if random.random() < self.prob:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        
        # 旋转 90/180/270 度
        if random.random() < self.prob:
            k = random.randint(1, 3)
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()
        
        # 亮度调整
        if random.random() < self.prob:
            alpha = random.uniform(0.7, 1.3)
            image = np.clip(image * alpha, 0, 255).astype(np.uint8)
        
        # 对比度调整
        if random.random() < self.prob:
            alpha = random.uniform(0.7, 1.3)
            image = np.clip((image - 128) * alpha + 128, 0, 255).astype(np.uint8)
        
        return image, mask

# ==================== 数据集 ====================

class AquaPlantDataset(Dataset):
    def __init__(self, root_dirs, img_size=1024, mode='train', use_augmentation=True):
        self.img_size = img_size
        self.mode = mode
        self.use_augmentation = use_augmentation and (mode == 'train')
        self.augmentation = SimpleAugmentation(prob=0.5) if self.use_augmentation else None
        self.data_list = []
        
        for root_dir in root_dirs:
            if os.path.exists(root_dir):
                for subdir in os.listdir(root_dir):
                    full_path = os.path.join(root_dir, subdir)
                    if os.path.isdir(full_path):
                        jpg_files = glob.glob(os.path.join(full_path, "*.jpg"))
                        if jpg_files:
                            self.data_list.append(full_path)
        
        aug_str = " (使用增强)" if self.use_augmentation else ""
        print(f"{mode} 数据集: {len(self.data_list)} 张影像{aug_str}")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        folder_path = self.data_list[idx]
        
        jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))
        mask_files = glob.glob(os.path.join(folder_path, "*_binaryMask.png"))
        
        image = cv2.imread(jpg_files[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_files[0], cv2.IMREAD_GRAYSCALE)
        
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        
        # 数据增强
        if self.augmentation:
            image, mask = self.augmentation(image, mask)
        
        # 正规化 - mask已经是0和1，不除以255！
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32)
        
        # 转 tensor
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask

# ==================== 损失函数 ====================

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        # BCE Loss (官方使用)
        bce_loss = self.bce(pred, target)
        
        # Dice Loss
        pred_sigmoid = torch.sigmoid(pred)
        smooth = 1e-5
        intersection = (pred_sigmoid * target).sum()
        union = pred_sigmoid.sum() + target.sum()
        dice_score = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice_score
        
        # 组合: BCE(60%) + Dice(40%)
        total = 0.6 * bce_loss + 0.4 * dice_loss
        return total, bce_loss, dice_loss

# ==================== 训练函数 ====================

def dice_coefficient(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    total_bce = 0
    total_dice_loss = 0
    total_dice = 0
    
    pbar = tqdm(dataloader, desc="训练")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss, bce, dice_loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        dice = dice_coefficient(outputs, masks)
        total_loss += loss.item()
        total_bce += bce.item()
        total_dice_loss += dice_loss.item()
        total_dice += dice.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice.item():.4f}'
        })
    
    n = len(dataloader)
    return total_loss/n, total_bce/n, total_dice_loss/n, total_dice/n

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="验证"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss, _, _ = criterion(outputs, masks)
            dice = dice_coefficient(outputs, masks)
            
            total_loss += loss.item()
            total_dice += dice.item()
    
    n = len(dataloader)
    return total_loss/n, total_dice/n

# ==================== 主程序 ====================

def main():
    base_dir = "/home/graduate/jinghan/lab/plant_v2/AqUavplant"
    
    # 训练集: BAU Botanical Garden, BAU Museum, Shapla Bil 1-4, Zinda Park 1-2
    train_dirs = [
        os.path.join(base_dir, "BAU Botanical Garden"),
        os.path.join(base_dir, "BAU Museum"),
        os.path.join(base_dir, "Shapla Bil 1"),
        os.path.join(base_dir, "Shapla Bil 2"),
        os.path.join(base_dir, "Shapla Bil 3"),  # 新增
        os.path.join(base_dir, "Shapla Bil 4"),
        os.path.join(base_dir, "Zinda Park 1"),
        os.path.join(base_dir, "Zinda Park 2"),
    ]
    
    # 验证集: Zinda Park 3
    val_dirs = [os.path.join(base_dir, "Zinda Park 3")]
    
    print("\n载入数据...")
    train_dataset = AquaPlantDataset(train_dirs, IMG_SIZE, 'train', use_augmentation=True)
    val_dataset = AquaPlantDataset(val_dirs, IMG_SIZE, 'val', use_augmentation=False)
    
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print("\n建立模型...")
    model = U_Net(img_ch=3, output_ch=1)
    
    if len(DEVICE_IDS) > 1:
        print(f"使用 {len(DEVICE_IDS)} 个GPU: {DEVICE_IDS}")
        model = nn.DataParallel(model, device_ids=DEVICE_IDS)
    model = model.to(DEVICE)
    
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    best_val_dice = 0
    patience_counter = 0
    
    print(f"\n开始训练（最多 {NUM_EPOCHS} epochs）")
    print("=" * 70)
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        train_loss, train_bce, train_dice_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler
        )
        val_loss, val_dice = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        
        print(f"训练 - Loss: {train_loss:.4f} (BCE: {train_bce:.4f}, Dice: {train_dice_loss:.4f}), Dice: {train_dice:.4f}")
        print(f"验证 - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
        
        # Early Stopping
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), 'best_unet_v2.pth')
            print(f"✓ 新最佳模型！Dice: {best_val_dice:.4f}")
        else:
            patience_counter += 1
            print(f"  未改善 ({patience_counter}/{EARLY_STOP_PATIENCE})")
        
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n🛑 Early Stop！已 {EARLY_STOP_PATIENCE} epochs 没有改善")
            break
        
        # 保存检查点
        if (epoch + 1) % 25 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_val_dice,
            }, f'checkpoint_v2_epoch_{epoch+1}.pth')
    
    elapsed = time.time() - start_time
    print(f"\n训练完成！耗时: {elapsed/3600:.2f} 小时")
    print(f"最佳 Dice: {best_val_dice:.4f}")
    
    # 保存训练日志
    with open('training_v2.log', 'w') as f:
        f.write("Epoch,Train_Loss,Train_Dice,Val_Loss,Val_Dice\n")
        for i in range(len(train_losses)):
            f.write(f"{i+1},{train_losses[i]:.6f},{train_dices[i]:.6f},{val_losses[i]:.6f},{val_dices[i]:.6f}\n")
    
    # 绘图
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train', alpha=0.8)
    plt.plot(val_losses, label='Val', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_dices, label='Train', alpha=0.8)
    plt.plot(val_dices, label='Val', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    plt.title(f'Dice Curve (Best: {best_val_dice:.4f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves_v2.png', dpi=300)
    print("✅ 训练曲线已保存")

if __name__ == "__main__":
    main()
