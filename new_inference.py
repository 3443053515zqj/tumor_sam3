"""
inference.py - 单张推理，同时保存纯掩码和原图掩码叠加
用法:
    python inference.py \
        --image ./test.jpg \
        --prompt malignant \
        --model_dir ./sam3_checkpoint/facebook/sam3 \
        --checkpoint best_model_epoch10_valloss0.0819.pth \
        --output_mask mask.png \
        --output_overlay overlay.png
"""

import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from model_utils import load_model_and_processor


def segment_and_overlay(model, processor, image_path, prompt,
                        image_size=256, device='cpu'):
    model.eval()
    model.to(device)

    # ---- 加载原始图像（保留原始尺寸用于叠加） ----
    original_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = original_img.size

    # ---- 预处理 ----
    img = original_img.resize((image_size, image_size), Image.BILINEAR)
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)

    text_encoding = processor(text=[prompt], return_tensors="pt",
                              padding=True, truncation=True)
    input_ids = text_encoding["input_ids"].to(device)
    attention_mask = text_encoding.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # ---- 推理 ----
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask)
        pred_masks = outputs.pred_masks
        if pred_masks.dim() == 4:
            pred_masks = pred_masks[:, 0, :, :]  # 取第一个掩码
        # 上采样到原始尺寸
        pred_masks = F.interpolate(pred_masks.unsqueeze(1),
                                   size=(orig_h, orig_w),
                                   mode='bilinear', align_corners=False)
        pred_mask = torch.sigmoid(pred_masks).squeeze().cpu().numpy()

    # ---- 生成二值掩码 ----
    binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    # ---- 生成叠加图（原图 + 半透明红色掩码） ----
    overlay = np.array(original_img).copy()
    # 将掩码区域染成红色（R=255, G=0, B=0），透明度 0.4
    mask_rgb = np.stack([binary_mask, np.zeros_like(binary_mask), np.zeros_like(binary_mask)], axis=-1)
    overlay = np.where(mask_rgb > 0,
                       (overlay * 0.6 + mask_rgb * 0.4).astype(np.uint8),
                       overlay)

    return binary_mask, overlay


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./sam3_checkpoint/facebook/sam3")
    parser.add_argument("--checkpoint", type=str, default="./best_model_epoch39_valloss0.0776.pth")
    parser.add_argument("--image", type=str, default='./data/busi/Dataset_BUSI_with_GT/benign/benign (10).png')
    parser.add_argument("--prompt", type=str, default="benign")
    parser.add_argument("--output_mask", type=str, default="mask.png")
    parser.add_argument("--output_overlay", type=str, default="overlay.png")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model_and_processor(args.model_dir)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print("模型加载成功！")

    mask, overlay = segment_and_overlay(
        model, processor, args.image, args.prompt,
        image_size=args.image_size, device=device
    )
    Image.fromarray(mask).save(args.output_mask)
    Image.fromarray(overlay).save(args.output_overlay)
    print(f"纯掩码已保存至 {args.output_mask}")
    print(f"原图叠加已保存至 {args.output_overlay}")
