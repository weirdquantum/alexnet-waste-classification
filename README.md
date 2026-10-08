# 基于 AlexNet 的垃圾图像分类 (AlexNet Waste Image Classification)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weirdquantum/alexnet-waste-classification/blob/main/notebooks/colab_run_all.ipynb)

本科课程 **UESTC 3002: Artificial Intelligence & Machine Learning**（2024–2025 秋季学期）的 Lab Project：用 PyTorch 实现 AlexNet，对 COCO 格式标注的垃圾图像做分类，并与传统机器学习基线（HOG + PCA + 决策树）对比。

**v2 版本**在原课程代码的基础上做了代码审查和重构：修复了数据泄漏、"最佳模型"保存逻辑、混淆矩阵方向不一致等问题；换用完整的 [TrashNet](https://github.com/garythung/trashnet) 数据集（2527 张，6 类），按类别分层划分训练/验证/测试集；对比了四种 AlexNet 训练配置和两个传统基线。原始课程代码保留在 [`legacy/`](legacy/)。

## 结果

> 完整结果待在 Colab GPU 上运行 [`notebooks/colab_run_all.ipynb`](notebooks/colab_run_all.ipynb) 后补充。

## 原版代码的问题

| # | 位置 | 问题 | 影响 |
|---|---|---|---|
| 1 | `train.py` / `evaluate.py` / notebook | 训练、验证、测试都读同一份 `annotations.json` | 数据泄漏，notebook 中的 100% 测试准确率没有意义 |
| 2 | `data_coco/` | 每类只有 1 张图，共 5 张 | 无法训练和评估 |
| 3 | `train.py` | `best_accuracy = 0` 写在 epoch 循环内部，每轮被重置 | 每轮都会覆盖保存，"best_model.pth" 实际是最后一轮 |
| 4 | `train.py` | 没有 `model.train()` / `model.eval()` 切换 | 验证时 Dropout 仍然开启，验证结果偏低且有随机性 |
| 5 | 原训练配方 | 输入只除以 255、无数据增强、无 BatchNorm、Adam lr=1e-3 从零训练 | notebook 日志中 loss 停在 1.85 左右（7 类随机猜测为 ln 7 ≈ 1.95），验证准确率在 0–40% 间波动，模型基本没有学到东西 |
| 6 | `model.py` / `ml_evaluate.py` | AlexNet 硬编码 7 类，数据只有 5 类；ML 基线又硬编码 5 类 | 类别数与数据不一致 |
| 7 | `evaluate.py` vs `ml_evaluate.py` | 前者 `matrix[pred, true]`（行=预测），后者 `matrix[true, pred]`（行=真实） | 两个混淆矩阵方向相反，无法直接对比 |
| 8 | `dataset.py` | 只把灰度图转成 RGB，RGBA / 调色板 / CMYK 图片会触发断言；bbox 没有裁剪到图像边界；打开的文件没有关闭 | 遇到非 RGB 图片或越界标注时崩溃 |
| 9 | 各脚本 | 数据路径写成 `data/data_coco`（实际是 `data_coco/`），`dataset.py` 和 `utils/` 中硬编码 `/root/projects/...` | 无法直接运行 |
| 10 | `submission/Data_Processing.py` | 从 annotation 读 `file_name`（COCO 中该字段在 `images` 里），且没有按 bbox 裁剪；路径 `'D:\Combined_COCO_Dataset\annotations.json'` 中的 `\a` 被 Python 解析成响铃字符 | `KeyError`，路径错误 |
| 11 | `submission/Model_Training.py` | 用测试集挑选最佳模型 | 测试结果偏乐观 |
| 12 | `submission/Performance_Evaluate.py` | 缺少 `import torch`，类别名是 `class_1`…`class_7` 占位符 | 无法单独运行，报告不可读 |
| 13 | `mlmodel.py` | PCA 只保留 5 维（受限于 5 个样本）；决策树没有设随机种子；`np.stack` 抛出的是 `ValueError`，代码却捕获 `RuntimeError` | 基线结果不可复现 |
| 14 | 评估 | 只看总体准确率；`torch.load` 未设置 `weights_only=True` | 类别不均衡时（trash 类只有 137 张）看不出小类表现；加载 checkpoint 有安全隐患 |
| 15 | 工程 | 同一份代码在 `.py`、notebook、`submission/` 中重复三遍，没有依赖列表、随机种子和测试 | 难以维护和复现 |

## 改进

- **数据**：完整 TrashNet 数据集（课程样例图片 `metal27.jpg` 等即来自 TrashNet），按类别分层划分 70 / 15 / 15，逐字节相同的重复图片（3 张）保证落在同一划分。标注统一转成 COCO 格式，数据集类仍然按 bbox 裁剪，和课程任务一致。
- **数据加载**：所有图片统一 `convert("RGB")`，bbox 裁剪到图像边界，类别 id 自动映射到 `0..K-1`，类别数从标注文件读取。
- **训练**：修复最佳模型保存与 train/eval 模式；按验证集 macro-F1 选模型并早停，测试集只在最后评估一次；数据增强（RandomResizedCrop、翻转、颜色扰动）、ImageNet 均值方差归一化、AdamW + label smoothing + warmup/cosine 学习率；支持 CUDA / Apple MPS / CPU。
- **模型**：同一个 `AlexNet` 类支持原版、加 BatchNorm、加载 ImageNet 预训练权重三种用法（参数名与 torchvision 一致）。
- **评估**：统一混淆矩阵方向（行=真实，列=预测），输出准确率、macro-F1、每类 precision / recall / F1，保存混淆矩阵和训练曲线图。
- **基线**：改成 sklearn Pipeline，PCA 只在训练集上拟合并保留 95% 方差；在验证集上调参；新增 RBF-SVM 对比。
- **工程**：`src/` 包 + 命令行脚本 + `pytest` 单元测试（数据裁剪、类别映射、划分、模型结构、指标）。

## 项目结构

```
.
├── src/wastecls/
│   ├── data.py              # CocoCropDataset（按 bbox 裁剪）、数据增强、分层划分
│   ├── models.py            # AlexNet / AlexNet-BN / ImageNet 预训练 AlexNet
│   ├── engine.py            # 训练与推理循环
│   ├── metrics.py           # 混淆矩阵、macro-F1、每类指标、绘图
│   ├── baseline.py          # HOG + 标准化 + PCA + 分类器 Pipeline
│   └── utils.py             # 随机种子、设备选择
├── scripts/
│   ├── prepare_trashnet.py  # TrashNet zip → COCO 标注 + train/val/test 划分
│   ├── train.py             # 训练 AlexNet（--preset course / scratch / scratch_bn / finetune）
│   ├── evaluate.py          # 评估已保存的 checkpoint
│   ├── train_baseline.py    # HOG + 决策树 / SVM 基线
│   └── summarize.py         # 汇总 results/*/metrics.json 为表格
├── notebooks/
│   └── colab_run_all.ipynb  # Colab 一键运行全部实验
├── tests/                   # pytest 单元测试
├── results/                 # 各实验的指标、混淆矩阵、训练曲线
└── legacy/                  # 原始课程代码（代码未改动，供对比）
    ├── experiment.ipynb     # 原实验 notebook（含当时的运行输出）
    ├── dataset.py  model.py  train.py  evaluate.py
    ├── mlmodel.py  ml_evaluate.py  outputs/model.pkl
    ├── data_coco/           # 课程样例数据（5 张图）
    ├── utils/
    └── submission/          # 随报告提交的分模块代码
```

## 运行

**Colab（推荐）**：点击上方 *Open in Colab*，切换到 GPU 运行时后全部运行。

**本地**：

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

```bash
mkdir -p data/download
curl -L -o data/download/dataset-resized.zip https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip
python scripts/prepare_trashnet.py
```

```bash
python scripts/train_baseline.py               # HOG + 决策树 / SVM
python scripts/train.py --preset course        # 原课程配方
python scripts/train.py --preset scratch       # AlexNet 从零训练，改进配方
python scripts/train.py --preset scratch_bn    # AlexNet-BN 从零训练
python scripts/train.py --preset finetune      # ImageNet 预训练 AlexNet 微调
python scripts/summarize.py                    # 生成 results/summary.md
```

命令行参数可以覆盖 preset 中的任意设置，例如 `--epochs 50 --lr 3e-4`。

## 局限

- TrashNet 中同一物体常以不同角度拍摄多张，但数据没有物体编号，无法按物体划分，测试集准确率可能偏高。
- 每张图只有一个物体、背景单一，模型在真实场景（杂乱背景、多物体）中的效果需要另行验证。
- 每个配置只用一个随机种子跑了一次，没有报告方差。

实验报告和课程讲义未包含在仓库中。
