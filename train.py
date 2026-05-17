"""
train.py - SAM3 DDP 训练脚本（BF16）
适配 多张GPU

用法:
    python train.py \
        --data_dir autodl-tmp/data/busi/Dataset_BUSI_with_GT \
        --batch_size 2 \
        --image_size 128 \
        --epochs 10
"""

import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
from tqdm import tqdm

from dataset import BUSIDataset
from model_utils import load_model_and_processor, save_checkpoint


def compute_loss(pred_masks, gt_masks):
    if pred_masks.dim() == 4:
        pred_masks = pred_masks[:, 0, :, :]
    pred_masks = pred_masks.unsqueeze(1)
    pred_masks = F.interpolate(pred_masks, size=gt_masks.shape[1:],
                               mode='bilinear', align_corners=False)
    return F.binary_cross_entropy_with_logits(pred_masks.float(), gt_masks.unsqueeze(1).float())


def train_ddp(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    model, processor = load_model_and_processor(args.model_dir)
    model.to(device)

    if hasattr(model, 'config'):
        model.config.use_cache = False

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    train_dataset = BUSIDataset(
        root_dir=args.data_dir,
        processor=processor,
        image_size=args.image_size,
        split="train"
    )
    val_dataset = BUSIDataset(
        root_dir=args.data_dir,
        processor=processor,
        image_size=args.image_size,
        split="val"
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=rank, shuffle=True, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=4,
                              pin_memory=True, drop_last=True)

    if rank == 0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss = 0.0

        if rank == 0:
            loop = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [GPU 0]')
        else:
            loop = train_loader

        for batch in loop:
            # 将图像直接转换为 BF16，节省显存
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            masks = batch["mask"].to(device, dtype=torch.float32)  # 掩码保持 FP32
            prompts = batch["text_prompt"]

            text_encoding = processor(text=prompts, return_tensors="pt",
                                      padding=True, truncation=True)
            input_ids = text_encoding["input_ids"].to(device)
            attention_mask = text_encoding.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            optimizer.zero_grad()

            outputs = model(pixel_values=pixel_values,
                            input_ids=input_ids,
                            attention_mask=attention_mask)
            loss = compute_loss(outputs.pred_masks, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if rank == 0:
                loop.set_postfix(loss=loss.item())

        avg_train = train_loss / len(train_loader)
        tensor = torch.tensor([avg_train]).to(device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        avg_train = tensor.item() / world_size

        if rank == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
                    masks = batch["mask"].to(device, dtype=torch.float32)
                    prompts = batch["text_prompt"]

                    text_encoding = processor(text=prompts, return_tensors="pt",
                                              padding=True, truncation=True)
                    input_ids = text_encoding["input_ids"].to(device)
                    attention_mask = text_encoding.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)

                    outputs = model(pixel_values=pixel_values,
                                    input_ids=input_ids,
                                    attention_mask=attention_mask)
                    loss = compute_loss(outputs.pred_masks, masks)
                    val_loss += loss.item()

            avg_val = val_loss / len(val_loader)
            print(f'Epoch {epoch}/{args.epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}')

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                save_checkpoint(model.module, f"best_model_epoch{epoch}_valloss{best_val_loss:.4f}.pth")
                print(f"保存最佳模型 (Val Loss: {best_val_loss:.4f})")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/busi/Dataset_BUSI_with_GT",
                        help="BUSI 数据集路径")
    parser.add_argument("--model_dir", type=str, default="./sam3_checkpoint/facebook/sam3",
                        help="SAM3 预训练模型路径")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(f"使用 {world_size} 张 GPU，每张 batch_size = {args.batch_size}，图像尺寸 = {args.image_size}，BF16 模式")
    mp.spawn(train_ddp, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
