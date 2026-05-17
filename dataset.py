"""
dataset.py - BUSI 数据集自动下载 + 数据加载
"""
import os
import zipfile
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# -----------------------------------------------------------------
# 自动下载部分 (使用 kagglehub, 若失败可手动下载)
# -----------------------------------------------------------------
def download_busi(download_root="./data"):
    """自动下载 BUSI 乳腺超声数据集 (约 2GB)"""
    save_path = Path(download_root) / "busi"
    if save_path.exists():
        print(f"数据集已存在: {save_path}")
        return str(save_path)

    print("正在下载 BUSI 数据集...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")
        # kagglehub 会下载到缓存目录，我们复制到指定位置
        import shutil
        shutil.copytree(path, save_path)
        print(f"数据集已下载至: {save_path}")
    except ImportError:
        print("错误: 需要安装 kagglehub。请运行: pip install kagglehub")
        print("或手动下载数据集并解压至 ./data/busi 目录")
        raise
    except Exception as e:
        print(f"自动下载失败: {e}")
        print("请手动从 https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset 下载")
        raise
    return str(save_path)


# -----------------------------------------------------------------
# 数据集类
# -----------------------------------------------------------------
class BUSIDataset(Dataset):
    def __init__(self, root_dir, processor, image_size=256, split="train"):
        self.root_dir = Path(root_dir)
        self.processor = processor
        self.image_size = image_size

        # 收集样本...
        self.samples = []
        categories = ["benign", "malignant", "normal"]
        for cat in categories:
            img_dir = self.root_dir / cat
            if not img_dir.exists():
                continue
            for fname in os.listdir(img_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')) and "_mask" not in fname:
                    mask_name = fname.rsplit('.', 1)[0] + "_mask." + fname.rsplit('.', 1)[1]
                    mask_path = img_dir / mask_name
                    self.samples.append({
                        "image_path": str(img_dir / fname),
                        "mask_path": str(mask_path) if mask_path.exists() else None,
                        "prompt": cat
                    })
        # 划分 train/val...
        n = len(self.samples)
        split_idx = int(n * 0.8)
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        inputs = self.processor(images=image, return_tensors="pt")

        if sample["mask_path"] and os.path.exists(sample["mask_path"]):
            mask = Image.open(sample["mask_path"]).convert("L")
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            mask = T.ToTensor()(mask)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "text_prompt": sample["prompt"],
            "mask": mask.squeeze(0),
        }

if __name__ == "__main__":
    save_path=download_busi()
