# tumor_sam3
基于SAM3的乳腺超声图像分割系统#医学深度学习

支持多模态输入，数据集直接运行dataset.py就能下载
大约运行50轮就可以达到最佳效果


# SAM3 乳腺超声图像分割

基于 Meta Segment Anything Model 3 (SAM3) 的乳腺超声肿瘤分割项目，支持文本提示驱动的病灶区域定位。项目使用 BUSI 数据集进行微调，提供数据预处理、分布式训练、推理与交互式 Web 界面，代码结构清晰，便于二次开发或教学演示。

## 环境要求

- Python 3.10+
- PyTorch 2.1+ (推荐 2.5+)
- CUDA 12.4+ (若使用 RTX 5090)
- 依赖库见下面 

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers modelscope gradio Pillow tqdm

## 数据集结构如下
data/busi/Dataset_BUSI_with_GT/
├── benign/
├── malignant/
└── normal/

##项目结构
.
├── dataset.py          # 数据集加载、预处理及自动下载
├── model_utils.py      # 从 ModelScope 下载 SAM3 权重、模型加载与保存
├── train.py            # 分布式训练脚本 (DDP，支持 4 卡 RTX 5090)
├── inference.py        # 单张图像推理，输出掩码与叠加图
├── app.py              # Gradio 交互式 Web 界面
└── README.md


执行命令：
python train.py \
    --data_dir ./data/busi/Dataset_BUSI_with_GT \
    --batch_size 2 \
    --epochs 10 \
    --image_size 256

python inference.py \
    --image ./test.jpg \
    --prompt malignant \
    --checkpoint best_model_epoch10_valloss0.0819.pth \
    --output_mask mask.png \
    --output_overlay overlay.png


##需要训练完之后的参数文件：
python app.py --checkpoint best_model_epoch10_valloss0.0819.pth

