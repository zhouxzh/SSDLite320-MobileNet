# MobileNet-SSDLite320 (PyTorch)

本项目是一个基于 `timm MobileNet` 主干网络的 `SSDLite320` 目标检测实现。代码结构按“教材可读性优先”整理，训练与验证流程面向 **多卡 GPU** 场景做了优化。

如果把整个项目当成教材来看，可以把训练与验证部分拆成三层：
- 命令行入口层：负责 `train / val` 模式切换、参数解析、DDP 初始化、断点恢复和导出 ONNX。
- 训练流程层：负责 warmup、学习率调度、验证、早停和 checkpoint。
- 模型与编码层：负责 MobileNet 主干、SSDLite 检测头、default boxes、编码解码。

这样拆分的目标不是“函数越多越好”，而是让每一层都回答一个明确问题：
- 入口层回答“训练和验证怎么启动”。
- 流程层回答“每个 epoch 具体做什么”。
- 模型层回答“网络和框编码是怎么定义的”。

## 1. 当前实现概览

### 模型
- 输入尺寸：`320 x 320`
- 检测头：`SSDLite` 风格，使用 `Depthwise 3x3 + Pointwise 1x1`
- 激活函数：`ReLU6`
- 归一化：`BatchNorm2d(eps=0.001, momentum=0.03)`
- 额外特征层：`1x1 -> depthwise 3x3(stride=2) -> 1x1`
- 初始化：卷积层使用 `normal_(mean=0, std=0.03)`
- 支持的主干：`MobileNet v1 / v2 / v3 / v4` 多个具体变体，便于做系统对比实验

对应代码：`ssdlite320/model.py`

### 先验框（Default Boxes）
- 特征层尺寸：`[20, 10, 5, 3, 2, 1]`
- 宽高比：每层 `[[2, 3]]`
- scale 范围：默认 `min_ratio=0.1`, `max_ratio=0.9`，可通过训练参数覆盖
- 生成方式：与 torchvision `DefaultBoxGenerator` 的 ratio/scale 思路对齐

对应代码：`ssdlite320/encoder.py` 中 `dboxes320_coco()`

### 训练策略（当前默认）
- 优化器：`SGD(momentum=0.9)` + Tencent trick（BN 和 bias 不做 weight decay）
- 学习率调度：采用 `freeze phase warmup + full phase warmup + cosine annealing` 的两阶段策略
- 混合精度：`torch.amp`（GPU 自动启用）
- Warmup：冻结阶段和全量训练阶段各自独立配置，互不混用
- 冻结策略：默认先冻结 backbone `5` 个 epoch（`--freeze-backbone-epochs 5`）
- Cosine 默认最小学习率比例：`0.02`，即最低降到有效学习率的 `2%`
- 默认 warmup：`--freeze-warmup-epochs 1`，`--warmup-epochs 3`
- 早停：`patience=20`，`min_delta=1e-4`；默认会根据余弦退火进入稳定阶段的时间自动选择合适的起始 epoch
- 每轮验证：默认 `eval_interval=1`
- 保存策略：
  - 续训保存：`checkpoints/ssd320_{backbone}_last.pth`，每轮覆盖更新
  - 最优保存：`checkpoints/ssd320_{backbone}_best.pth`

对应代码：`ssdlite320/train.py`

### 一轮训练是怎么组织的
当前默认配置下，训练流程可以直接理解成下面这条主线：

1. 先根据 batch size 和 world size 计算线性缩放后的有效学习率。
2. 构建 `SSD320`、`Loss`、`Encoder` 和 TensorBoard writer。
3. 如果启用了 `--restart`，优先从 last checkpoint 恢复模型权重。
4. 如果 `--freeze-backbone-epochs > 0`，先运行“冻结 backbone、只训练检测头”的阶段。
5. 进入全量训练阶段，对 backbone 和检测头一起优化。
6. 如果显式设置了 `--freeze-warmup-epochs` 或 `--warmup-epochs`，会先做标准 `LinearLR` warmup。
7. warmup 结束后，训练按标准 PyTorch `CosineAnnealingLR` 继续衰减学习率。
8. 每隔 `--eval-interval` 个 epoch 在验证集上计算 COCO mAP，并保存可视化结果。
9. 如果 mAP 创新高，就写入 `best` checkpoint；如果长期无提升，就触发 early stopping。

这个顺序与代码中的主入口、训练上下文和阶段函数是一一对应的，适合顺着代码讲课。

### 训练代码结构
- `main.py`：统一入口，负责参数定义、参数校验、`train / val` 命令分发。
- `ssdlite320/runtime.py`：承接 DDP 初始化与销毁、数据加载、验证资源构建、checkpoint 查找和 ONNX 导出等工程性辅助逻辑。
- `ssdlite320/train.py`：只保留“训练过程本身”的逻辑，例如优化器构建、epoch 训练、验证、保存、早停。
- `ssdlite320/eval.py`：统一放 PyTorch 验证与 ONNX 验证逻辑，包括可视化、COCO 指标计算和 CSV 导出。
- `ssdlite320/utils.py`：集中放共享的可视化、类别名解析和验证指标整理逻辑。
- `ssdlite320/data_hf.py`：把“数据下载、标注解析、训练增强、验证缩放”拆成独立接口，避免把数据细节堆在 `__getitem__` 里。
- `ssdlite320/model.py`：把“backbone 名称映射、额外特征层、预测头、loss”拆成几块稳定组件，便于单独讲解。
- `TrainingContext`：集中保存跨阶段共享的运行时对象，避免把 `model / writer / criterion / device` 塞进零散字典中。
- `ssdlite320/encoder.py`：只负责 default boxes、编码和解码，不再混入训练可视化逻辑。

这套划分有一个明确标准：
- 如果一个辅助函数只是机械地包了一行表达式，而且没有提供额外语义，一般直接内联。
- 如果一个辅助函数封装的是“接口契约”或“训练阶段语义”，则保留成独立模块或独立函数。

---

## 2. 环境要求

建议使用 conda 创建独立环境（如 `torch`）以便管理依赖：

```bash
conda create -n torch
```

### 激活环境

```bash
conda activate torch
```

### 安装 Python

推荐安装 Python 3.13：

```bash
conda install python=3.13
```

### 安装 PyTorch 与 TorchVision

请根据你的 CUDA 驱动版本，参考 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取对应安装命令。以 CUDA 13.0 和 PyTorch 2.11.0 为例：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

> 请根据实际 CUDA 版本调整 `cu126` 或者 `cuda128` 部分。

### 安装项目依赖

```bash
pip install -r requirements.txt
```

---

## 3. 训练命令

统一入口：`main.py`

建议先把命令行参数理解成 5 组：
- model：backbone、类别数、是否加载预训练骨干。
- training strategy：freeze、warmup、epoch、验证频率、验证可视化数量。
- optimization：学习率、余弦最小学习率比例、动量、weight decay、early stopping。
- data loading：worker、prefetch、pin memory、augmentation。
- runtime and export：device、DDP、resume、checkpoint 恢复、导出 ONNX。

### 默认训练（单卡）
```bash
python main.py train --device cuda
```

这条命令当前默认等价于：`--epochs 300 --freeze-backbone-epochs 5 --freeze-warmup-epochs 1 --warmup-epochs 3 --cosine-min-lr-ratio 0.02 --patience 20 --num-visualizations 0`

TensorBoard 日志目录会自动带上 cosine 标签，例如 `logs/.../cosine_minlr_0p020`，方便直接对比。

训练模式默认会在结束后导出 ONNX。
- 默认导出训练结束时当前内存模型对应的 ONNX。
- 如果需要显式指定导出来源，可使用 `--export-onnx-from-best-checkpoint`。

### 从 last checkpoint 续训
```bash
python main.py train --device cuda --restart
```

### 快速 smoke test（先跑 10 轮）
```bash
python main.py train --device cuda --epochs 10 --patience 3
```

### 训练时开启验证集可视化
```bash
python main.py train --device cuda --num-visualizations 20
```

默认值是 `--num-visualizations 0`，也就是训练过程中默认不导出验证图片。

### 多卡训练示例（单节点，2 GPUs — 2x4090D）

在单台机器上使用 2 块 4090D（两个 GPU）运行 DDP，请使用 `torchrun` 或 `python -m torch.distributed.run`。示例命令（每卡进程数 = 1）：

```bash
# 使用 torchrun（推荐）
torchrun --nproc_per_node=2 main.py train --device cuda

# 等价的 Python 启动方式
python -m torch.distributed.run --nproc_per_node=2 main.py train --device cuda
```

说明：
- `--nproc_per_node=2` 表示每台机器启动 2 个进程（每个进程绑定一个 GPU）。
- `--batch-size` 是每进程 batch size；实际学习率会按 `batch_size * world_size` 做线性缩放。
- 如需跨多节点训练，请设置 `--nnodes`、`--node_rank`、`MASTER_ADDR` 和 `MASTER_PORT` 环境变量（示例见下）：

```bash
MASTER_ADDR=master_ip MASTER_PORT=12345 torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 main.py train --device cuda --ddp
```


---

### ONNX 验证并导出 CSV
```bash
python main.py val --backbone mobilenetv4_conv_small --provider auto --csv-file reports/onnx_validation_metrics.csv
```

这条命令会：
- 读取 `weights/ssd320_{backbone}.onnx`
- 优先复用 `data/coco_gt.json` 作为 COCO Ground Truth 缓存
- 固定在 COCO 验证集上计算 `mAP / AP50 / AP75 / small / medium / large`
- 把结果追加写入 `CSV`，方便比较不同导出模型

`val` 模式不提供 `dataset-name` 参数，验证数据集固定为 COCO val，保证结果可重复对比。

如需显式指定 ONNX 文件：
```bash
python main.py val --onnx-path weights/ssd320_mobilenetv4_conv_small.onnx --provider cuda --num-visualizations 20
```

## 4. 当前默认参数（main.py train）

- `--batch-size 64`
- `--epochs 300`
- `--lr 1e-3`
- `--momentum 0.9`
- `--weight-decay 4e-5`
- `--num-workers 8`
- `--warmup-epochs 3`
- `--freeze-backbone-epochs 5`
- `--freeze-warmup-epochs 1`
- `--cosine-min-lr-ratio 0.02`
- `--patience 20`
- `--min-delta 1e-4`
- `--eval-interval 1`
- `--num-visualizations 0`
- `--num-classes 81`（COCO 80 类 + background）
- `--pretrained-backbone` 默认开启（使用 timm 预训练权重）
- `--backbone mobilenetv4_conv_small`
- `--dbox-min-ratio 0.1`
- `--dbox-max-ratio 0.9`

### 这几个参数最值得先理解
- `--warmup-epochs`：控制全量训练阶段先做多少个标准 `LinearLR` warmup epoch，默认是 3。
- `--freeze-backbone-epochs`：决定是否启用“先只训检测头”的前置阶段。
- `--freeze-warmup-epochs`：只在冻结阶段生效，默认是 1，让开头几轮更稳。
- `--cosine-min-lr-ratio`：控制 `CosineAnnealingLR` 尾部的最小学习率比例。
- `--dbox-min-ratio` / `--dbox-max-ratio`：控制 default boxes 的整体 scale 覆盖范围。

---

## 参数推荐表（双卡冲榜）

| 场景 | 建议参数 | 说明 |
|---|---|---|
| 当前默认配置 | `--batch-size 64 --epochs 300 --lr 1.0e-3 --freeze-backbone-epochs 5 --freeze-warmup-epochs 1 --warmup-epochs 3 --cosine-min-lr-ratio 0.02 --patience 20 --num-workers 8` | 适合直接运行的默认配置，前期更平滑，后期继续走标准余弦退火 |
| 更稳的长训配置 | `--batch-size 64 --epochs 400 --lr 1.0e-3 --freeze-backbone-epochs 5 --freeze-warmup-epochs 1 --warmup-epochs 3 --cosine-min-lr-ratio 0.02 --patience 20 --num-workers 8` | 保留相同策略，只把总训练轮数拉长，适合继续追 mAP |
| 想让前期下降更早 | `--batch-size 64 --epochs 300 --lr 1.0e-3 --freeze-backbone-epochs 5 --freeze-warmup-epochs 0 --warmup-epochs 0 --cosine-min-lr-ratio 0.02 --patience 20 --num-workers 8` | 关闭 warmup 后，学习率会从一开始就进入余弦退火 |

如果目标是冲榜，建议先从“更稳的长训配置”开始。

示例命令（当前推荐的 cosine + 冻结 backbone 策略）：
```bash
torchrun --nproc_per_node=2 main.py train --device cuda --freeze-backbone-epochs 5 --freeze-warmup-epochs 1 --warmup-epochs 3 --cosine-min-lr-ratio 0.02 --epochs 300 --patience 20
```

## 5. 导出 ONNX

训练模式默认会在结束后自动导出 ONNX：
- `weights/ssd320_{backbone}.onnx`
- `mobilenetv4_hybrid_*` 系列会自动切到 opset 14，以支持 `scaled_dot_product_attention`

目录约定：
- `weights/`：保存导出的 ONNX 模型及相关推理文件。
- `checkpoints/`：只保存 `last` checkpoint 和 `best` checkpoint。

如果只想切换导出来源，可使用 `--export-onnx-from-best-checkpoint`，让导出基于 best checkpoint。

## 6. COCO Ground Truth 缓存

训练期验证和 `val` 模式都会使用 `data/coco_gt.json` 作为 COCO Ground Truth 缓存。

- 首次运行评估时，会自动根据当前 COCO val 数据集生成这个文件。
- 之后如果文件已经存在，就直接复用，不会每次重新计算。
- 如果读取这个文件时报错，程序会打印错误信息，并提示你手动删除 `data/coco_gt.json` 后重试。

这样做的目的是减少每次评估前重复构建 COCO Ground Truth 的时间开销。

---

## 7. 常见问题

### Q1: 想更快收敛，怎么调？
- 保持 `--pretrained-backbone` 开启
- 如果目标是冲榜，优先直接把 `epochs` 设到 `300` 或 `400`
- 如果 `300+ epoch` 后仍在缓慢上涨，优先增加总 epoch，或把 `--cosine-min-lr-ratio` 再调低一点
- 如果你希望学习率更早开始下降，可以把 `--freeze-warmup-epochs` 和 `--warmup-epochs` 调小

### Q2: 显存不足？
- 先降 `--batch-size` 到 `24` 或 `16`
- 保持 AMP（默认 GPU 已启用）

### Q3: 如何查看训练效果？
- TensorBoard 日志目录：`logs/{backbone}/min_xx_max_xx_{schedule_tag}`
- 训练或 ONNX 验证可视化结果：仅在 `--num-visualizations > 0` 时输出到 `viz_results/...`
- ONNX 验证结果 CSV：`reports/onnx_validation_metrics.csv`

### Q4: 为什么现在会删掉一些只有一行的辅助函数？
因为“短函数”本身不是问题，问题在于它是否传递了新的语义。

例如下面两类函数通常不值得单独保留：
- 只是把一个布尔表达式换个名字，但调用点并没有因此更清楚。
- 只是把一个字符串格式化语句包起来，但项目里也没有复用需求。

相反，下面两类函数通常值得保留：
- 明确表达训练阶段语义，例如“训练一个 epoch”“按需做验证”。
- 封装稳定接口契约，例如“构建参数分组”“构建训练上下文”。

教材代码不是函数越碎越好，而是抽象边界要让读者一眼看出职责。

---

## 8. 主要代码文件

建议阅读顺序：
1. 先看 `main.py`，理解整个训练/验证入口如何组织。
2. 再看 `ssdlite320/train.py`，重点看两个训练阶段是怎样复用同一套 epoch 执行流程的。
3. 然后看 `ssdlite320/eval.py`，把训练期验证和 ONNX 验证当成同一个“预测 -> COCO 指标”流程来理解。
4. 最后看 `ssdlite320/data_hf.py` 和 `ssdlite320/runtime.py`，补齐数据与工程辅助层细节。

- `main.py`：统一训练/验证入口，参数、校验和命令分发都在这里，适合初学者先看
- `ssdlite320/runtime.py`：DDP、DataLoader、验证资源、checkpoint 和导出等工程辅助逻辑
- `ssdlite320/train.py`：训练循环、调度、早停、checkpoint 保存
- `ssdlite320/model.py`：MobileNet + SSDLite320 模型定义
- `ssdlite320/encoder.py`：编码解码、IoU、Default Boxes
- `ssdlite320/data_hf.py`：COCO 数据集封装、数据增强与 DataLoader 构建
- `ssdlite320/utils.py`：共享工具函数，例如可视化、类别名解析、验证指标整理
- `ssdlite320/eval.py`：统一评估层，包含训练期验证和 ONNX 验证
