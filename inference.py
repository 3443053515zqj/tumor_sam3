"""
inference.py - 单张图片推理（支持 SAM3 文本提示）
用法:
    python inference.py \
        --image ./test.jpg \
        --prompt malignant \
        --model_dir ./sam3_checkpoint/facebook/sam3 \
        --checkpoint best_model_epoch10_valloss0.0819.pth \
        --output output_mask.png
"""

import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from model_utils import load_model_and_processor

def segment(model, processor, image_path, prompt, image_size=256, device='cpu'):
    model.eval()
    model.to(device)

    # 加载图像并转换为 RGB
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)

    # 处理器获取 pixel_values
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)

    # 文本转 input_ids
    text_encoding = processor(text=[prompt], return_tensors="pt", padding=True, truncation=True)
    input_ids = text_encoding["input_ids"].to(device)
    attention_mask = text_encoding.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask)
        pred_masks = outputs.pred_masks  # [B, N, H, W] 或 [B, H, W]
        if pred_masks.dim() == 4:
            pred_masks = pred_masks[:, 0, :, :]  # 取第一个掩码

        # 上采样到原始图像尺寸
        pred_masks = F.interpolate(pred_masks.unsqueeze(1),
                                   size=img.size[::-1],
                                   mode='bilinear', align_corners=False)
        pred_mask = torch.sigmoid(pred_masks).squeeze().cpu().numpy()

    # 二值化
    binary_mask = (pred_mask > 0.5).astype('uint8') * 255
    return binary_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./sam3_checkpoint/facebook/sam3",
                        help="SAM3 预训练模型目录")
    parser.add_argument("--checkpoint", type=str,default="./best_model_epoch39_valloss0.0776.pth",
                        help="训练好的权重文件（.pth）")
    parser.add_argument("--image", type=str, default='./data/busi/Dataset_BUSI_with_GT/benign/benign (10).png',
                        help="输入图像路径")
    parser.add_argument("--prompt", type=str, default="benign",
                        help="文本提示（benign, malignant, normal）")
    parser.add_argument("--output", type=str, default="output_mask.png",
                        help="输出掩码图像路径")
    parser.add_argument("--image_size", type=int, default=256,
                        help="训练时使用的图像尺寸，默认 256")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model_and_processor(args.model_dir)

    # 加载训练好的权重
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print("模型权重加载成功！")

    mask = segment(model, processor, args.image, args.prompt,
                   image_size=args.image_size, device=device)
    Image.fromarray(mask).save(args.output)
    print(f"分割结果已保存至 {args.output}")
