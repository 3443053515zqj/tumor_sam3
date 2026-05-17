"""
model_utils.py - 从 ModelScope 下载 SAM3 模型，并加载模型/处理器
"""
from modelscope import snapshot_download
from transformers import Sam3Model, Sam3Processor
import torch

def download_sam3_model(cache_dir="./sam3_checkpoint"):
    """从 ModelScope 下载 SAM3 预训练权重 (国内可访问)"""
    print("正在从 ModelScope 下载 SAM3 模型...")
    model_dir = snapshot_download("facebook/sam3", cache_dir=cache_dir)
    print(f"模型已下载至: {model_dir}")
    return model_dir

def load_model_and_processor(model_dir):
    """加载模型和处理器"""
    processor = Sam3Processor.from_pretrained(model_dir)
    model = Sam3Model.from_pretrained(model_dir)
    return model, processor

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    return model

if __name__ == '__main__':
    model_dir=download_sam3_model()
