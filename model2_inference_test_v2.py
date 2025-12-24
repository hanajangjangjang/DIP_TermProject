"""
使用训练好的 best_unet_v2.pth 对测试集进行推理
测试集: A1_20250402 至 A1_20250506 (cropped_focus)
"""

import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from tqdm import tqdm
import glob

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

# ==================== 推理函数 ====================

def inference_test_set(model_path, input_folder, output_folder, threshold=0.3, device='cuda'):
    """对测试集进行推理"""
    
    os.makedirs(output_folder, exist_ok=True)
    
    # 载入模型
    print("载入模型...")
    model = U_Net(img_ch=3, output_ch=1)
    state_dict = torch.load(model_path, map_location=device)
    
    # 移除 DataParallel 的 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    print(f"模型已载入，使用阈值: {threshold}")
    
    # 获取测试图片 A1_20250402 到 A1_20250506
    all_files = sorted(glob.glob(os.path.join(input_folder, "A1_*_masked.png")))
    image_files = [f for f in all_files if any(date in f for date in 
                   [f'202504{i:02d}' for i in range(2, 31)] + 
                   [f'202505{i:02d}' for i in range(1, 7)])]
    
    print(f"\n找到 {len(image_files)} 张测试图片")
    
    if len(image_files) == 0:
        print("❌ 没有找到测试图片！")
        return
    
    print("\n测试图片列表:")
    for img_file in image_files:
        print(f"  - {os.path.basename(img_file)}")
    
    print(f"\n开始推理...")
    print("=" * 70)
    
    coverage_list = []
    
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="处理中"):
            # 读取图片
            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠️  无法读取: {os.path.basename(img_path)}")
                continue
            
            original_image = image.copy()
            h, w = image.shape[:2]
            
            # 找出黑色区域（mask区域）
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            black_mask = (gray < 10)
            valid_area = (~black_mask).sum()
            
            # 预处理
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (1024, 1024))
            image_tensor = torch.from_numpy(image_resized.astype(np.float32) / 255.0)
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            
            # 推理
            output = model(image_tensor)
            pred = torch.sigmoid(output).cpu().numpy()[0, 0]
            
            # 调整回原始大小
            pred_resized = cv2.resize(pred, (w, h))
            pred_mask = (pred_resized > threshold).astype(np.uint8)
            
            # 在黑色区域不预测
            pred_mask[black_mask] = 0
            
            # 计算覆盖率
            plant_area = pred_mask.sum()
            coverage = (plant_area / valid_area * 100) if valid_area > 0 else 0
            coverage_list.append((os.path.basename(img_path), coverage))
            
            # 创建红色覆蓋层
            overlay = original_image.copy()
            overlay[pred_mask == 1] = [0, 0, 255]  # BGR: 红色
            
            # 混合 (alpha=0.5)
            result = cv2.addWeighted(original_image, 0.5, overlay, 0.5, 0)
            
            # 保存结果
            output_path = os.path.join(output_folder, os.path.basename(img_path))
            cv2.imwrite(output_path, result)
    
    print(f"\n✅ 完成！所有结果已保存到: {output_folder}")
    print(f"   总共处理: {len(image_files)} 张图片")
    
    # 显示覆盖率统计
    if coverage_list:
        coverages = [c for _, c in coverage_list]
        print(f"\n📊 预测覆盖率统计:")
        print(f"   平均: {np.mean(coverages):.2f}%")
        print(f"   标准差: {np.std(coverages):.2f}%")
        print(f"   范围: {min(coverages):.2f}% - {max(coverages):.2f}%")
        
        print(f"\n各图片覆盖率:")
        for name, cov in coverage_list:
            print(f"   {name:30s}: {cov:6.2f}%")

# ==================== 主程序 ====================

def main():
    MODEL_PATH = "best_unet_v2.pth"
    INPUT_FOLDER = "cropped_focus"
    OUTPUT_FOLDER = "results_test_v2"
    THRESHOLD = 0.5  # 官方默认阈值
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 70)
    print("🌿 测试集推理 - V2 模型")
    print("=" * 70)
    print(f"模型: {MODEL_PATH}")
    print(f"输入资料夹: {INPUT_FOLDER}")
    print(f"输出资料夹: {OUTPUT_FOLDER}")
    print(f"测试范围: A1_20250402 至 A1_20250506")
    print(f"阈值: {THRESHOLD}")
    print(f"装置: {DEVICE}")
    print("=" * 70)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型: {MODEL_PATH}")
        return
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ 找不到输入资料夹: {INPUT_FOLDER}")
        return
    
    inference_test_set(MODEL_PATH, INPUT_FOLDER, OUTPUT_FOLDER, THRESHOLD, DEVICE)

if __name__ == "__main__":
    main()
