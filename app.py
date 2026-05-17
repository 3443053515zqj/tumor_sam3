"""
app.py - Gradio Web 界面
用法:
    python app.py \
        --model_dir ./sam3_checkpoint/facebook/sam3 \
        --checkpoint best_model_epoch10_valloss0.0819.pth
"""

import argparse
import gradio as gr
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
from model_utils import load_model_and_processor

class SAM3MedDemo:
    def __init__(self, model_dir, checkpoint_path, image_size=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.processor = load_model_and_processor(model_dir)

        # 加载训练权重
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.image_size = image_size

    def predict(self, image, prompt):
        if image is None:
            return None
        # 预处理
        img = Image.fromarray(image).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=torch.bfloat16)

        text_encoding = self.processor(text=[prompt], return_tensors="pt",
                                       padding=True, truncation=True)
        input_ids = text_encoding["input_ids"].to(self.device)
        attention_mask = text_encoding.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values,
                                input_ids=input_ids,
                                attention_mask=attention_mask)
            pred_masks = outputs.pred_masks
            if pred_masks.dim() == 4:
                pred_masks = pred_masks[:, 0, :, :]
            # 上采样到原始尺寸
            pred_masks = F.interpolate(pred_masks.unsqueeze(1),
                                       size=(image.shape[0], image.shape[1]),
                                       mode='bilinear', align_corners=False)
            pred_mask = torch.sigmoid(pred_masks).squeeze().cpu().numpy()

        mask = (pred_mask > 0.5).astype(np.uint8) * 255
        return Image.fromarray(mask)

def main(args):
    demo = SAM3MedDemo(args.model_dir, args.checkpoint, image_size=args.image_size)
    iface = gr.Interface(
        fn=demo.predict,
        inputs=[
            gr.Image(type="numpy", label="上传乳腺超声图像"),
            gr.Textbox(value="benign", label="文本提示 (benign, malignant, normal)")
        ],
        outputs=gr.Image(type="pil", label="分割结果"),
        title="SAM3 乳腺超声肿瘤分割",
        description="上传乳腺超声图像，输入类别提示（如 benign、malignant），获得肿瘤区域分割掩码"
    )
    iface.launch(server_port=args.port, share=args.share)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./sam3_checkpoint/facebook/sam3")
    parser.add_argument("--checkpoint", type=str, default="./best_model_epoch39_valloss0.0776.pth")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--share", action="store_true", help="生成公网分享链接")
    args = parser.parse_args()
    main(args)
