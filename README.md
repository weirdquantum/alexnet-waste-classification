# 基于 AlexNet 的垃圾图像分类 (AlexNet Waste Image Classification)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weirdquantum/alexnet-waste-classification/blob/main/notebooks/colab_run_all.ipynb)
[![tests](https://github.com/weirdquantum/alexnet-waste-classification/actions/workflows/tests.yml/badge.svg)](https://github.com/weirdquantum/alexnet-waste-classification/actions/workflows/tests.yml)

本科课程 **UESTC 3002: Artificial Intelligence & Machine Learning**（2024–2025 秋季学期）的 Lab Project：用 PyTorch 实现 AlexNet，对 COCO 格式标注的垃圾图像做分类，并与传统机器学习基线（HOG + PCA + 决策树）对比。

**v2 版本**在原课程代码的基础上做了代码审查和重构：修复了数据泄漏、"最佳模型"保存逻辑、混淆矩阵方向不一致等问题；换用完整的 [TrashNet](https://github.com/garythung/trashnet) 数据集（2527 张，6 类），按类别分层划分训练/验证/测试集；对比了四种 AlexNet 训练配置和两个传统基线。原始课程代码保留在 [`legacy/`](legacy/)。

**v3 版本**修复了 v2 自查发现的问题（见[下文](#v2-自查发现的问题v3-已修复)）：删除标签冲突的图片、按物体分组划分以消除近重复图片泄漏、BN 和偏置不加权重衰减、每个配置跑 3 个随机种子；新增 Grad-CAM、单图预测脚本、端到端测试和 CI。

> **v3 的实验正在 Colab 上重新运行。** 下面的结果与发现来自 v2（旧划分，未去除近重复图片，单个随机种子），新结果出来后会替换。

**v2 最佳结果（ImageNet 预训练 AlexNet 微调）：测试集准确率 88.4%，macro-F1 86.6%。** 同样的 AlexNet 用原课程训练配方只有 23.5%（全部预测成 paper 一类）；只改训练配方、不用预训练，从零训练也能达到 83.6%。

![Confusion matrix of the fine-tuned AlexNet](results/finetune/confusion_matrix.png)

## 结果（v2）

TrashNet 测试集 378 张（cardboard 60 / glass 75 / metal 61 / paper 89 / plastic 72 / trash 21），在 Colab T4 GPU 上运行。每个配置按验证集 macro-F1 选最佳 epoch，测试集只在最后评估一次。数值为百分比。

| 模型 | 测试准确率 | 测试 macro-F1 | 验证 macro-F1 | 最佳 epoch | 训练时间 (min) |
|---|---|---|---|---|---|
| HOG + PCA + 决策树（原课程基线） | 33.3 | 31.6 | 29.4 | – | 2.3 |
| HOG + PCA + RBF-SVM | 67.7 | 66.9 | 62.3 | – | 1.5 |
| AlexNet，原课程配方 (`course`) | 23.5 | 6.4 | 6.3 | 23 / 30 | 2.0 |
| AlexNet 从零训练，改进配方 (`scratch`) | 83.6 | 82.3 | 86.6 | 99 / 100 | 16.1 |
| AlexNet-BN 从零训练，改进配方 (`scratch_bn`) | 79.9 | 77.1 | 82.9 | 73 / 98 | 15.5 |
| **AlexNet ImageNet 预训练微调 (`finetune`)** | **88.4** | **86.6** | 91.2 | 30 / 30 | 4.8 |

每类测试集 F1：

| 模型 | cardboard | glass | metal | paper | plastic | trash |
|---|---|---|---|---|---|---|
| HOG + PCA + 决策树 | 39.3 | 36.6 | 29.2 | 41.1 | 24.0 | 19.0 |
| HOG + PCA + RBF-SVM | 72.3 | 58.8 | 62.9 | 79.8 | 62.9 | 64.7 |
| AlexNet，原课程配方 | 0.0 | 0.0 | 0.0 | 38.1 | 0.0 | 0.0 |
| AlexNet 从零训练 | 90.3 | 77.0 | 80.3 | **92.8** | 79.5 | **73.9** |
| AlexNet-BN 从零训练 | 88.7 | 73.4 | 81.2 | 89.7 | 75.2 | 54.2 |
| AlexNet 预训练微调 | **92.2** | **85.0** | **90.0** | 91.8 | **87.9** | 72.7 |

各实验的指标、混淆矩阵和训练曲线在 [`results/`](results/) 下，汇总表由 `scripts/summarize.py` 生成。

### 主要发现

1. **原课程配方训练不起来。** 输入只除以 255、无数据增强、Adam lr=1e-3 从零训练 AlexNet，第 2 轮起训练 loss 就停在 1.725，正好等于训练集类别先验分布的熵（1.723）：模型只学到了"猜最多的那一类"，所有测试图片都预测为 paper（占 23.5%），比 HOG + 决策树还差。原 notebook 里 loss 停在 1.85 左右是同一个现象，只是被 5 张图的数据集和数据泄漏掩盖了。
2. **训练配方比网络结构更重要。** 同一个 AlexNet，只换成 ImageNet 均值方差归一化、数据增强、AdamW（lr 1e-4）+ warmup/cosine 学习率和 label smoothing，测试准确率从 23.5% 提高到 83.6%。
3. **加 BatchNorm 收敛更快，但最终结果没有更好。** AlexNet-BN 第 1 轮验证准确率就有 40.6%（不加 BN 为 23.0%），但最终测试准确率 79.9%，低于不加 BN 的 83.6%。它的训练准确率最后只有约 86%，仍在上升，说明这组超参数（lr 1e-3、weight decay 0.05）下是欠拟合，而不是过拟合。由于只跑了一个随机种子，3–4 个百分点的差距不足以说明 BN 本身更差。
4. **预训练带来的提升集中在易混淆的材质上。** 微调模型比从零训练高 4.8 个百分点，训练时间只有约 1/3。提升最大的是 glass（+8.0 F1）、plastic（+8.4）和 metal（+9.7）：透明、反光的材质，正好依赖 ImageNet 学到的纹理和光泽特征；paper、cardboard 两个模型都已经很高。剩下的主要错误是 cardboard → paper（7 张）以及 glass / metal / plastic 之间的互相混淆。
5. **trash 类最难。** 训练集只有 96 张，类内差异也最大（各种杂物），所有模型的 trash F1 都最低或接近最低。测试集只有 21 张，一两张图的差别就会让 F1 变化 5 个百分点以上，所以这一列的模型间差异不太可靠。
6. **传统基线中分类器的选择影响很大。** 同样的 HOG + PCA（515 维，保留 95% 方差）特征，决策树只有 33.3%，换成 RBF-SVM 就到 67.7%。决策树在高维连续特征上很容易过拟合，原课程基线的弱主要在分类器而不在特征。
7. **两个最好的模型在训练结束时仍在提升**（最佳 epoch 分别为 99/100 和 30/30），延长训练可能还有小幅提升。

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

## v2 自查发现的问题（v3 已修复）

| # | 问题 | 证据 | 修复 |
|---|---|---|---|
| 1 | **标签冲突**：TrashNet 中有 3 对逐字节相同的图片被放在两个不同类别下（`glass115` = `metal91`，`glass176` = `plastic152`，`glass389` = `plastic332`）。v2 只保证重复图片落在同一划分，没有检查标签，6 张图都在训练集里带着相互矛盾的标签 | MD5 相同 | 删除全部 6 张（无法确定真实类别） |
| 2 | **近重复图片泄漏**：同一物体常被拍多张，v2 随机划分会把它们分到训练集和测试集两边 | 用 ImageNet AlexNet fc6 特征计算余弦相似度：378 张测试图中有 18 张与某张训练图相似度 ≥ 0.9，34 张 ≥ 0.85（如 `metal76` / `metal394` 为 0.99，`metal172` / `metal173` 是同一个铝盘） | 同类且相似度 ≥ 0.85 的图片用并查集归为一组，按组分层划分。阈值取自无关同类图片对相似度的 99.9% 分位数（0.81）之上。划分后训练集与测试集之间同类最大相似度为 0.848 |
| 3 | BatchNorm 参数和偏置也加了 weight decay（0.05） | AlexNet-BN 训练准确率最终只有约 86%，欠拟合 | 只对卷积和全连接层权重做 weight decay |
| 4 | 只跑了一个随机种子，3–4 个百分点的差距无法判断是否显著 | – | 每个配置 3 个种子，报告均值 ± 标准差 |
| 5 | `course` 配置用的是 Kaiming 初始化，与原课程代码的 PyTorch 默认初始化不同 | – | 新增 `init` 选项，`course` 改用默认初始化 |
| 6 | 只有单元测试，训练、评估、数据准备脚本没有被测试覆盖；`results/` 为空时 `summarize.py` 报错 | – | 新增端到端测试（在合成数据上跑完整流程）和 GitHub Actions CI；`summarize.py` 给出明确提示 |

## 改进

- **数据**：完整 TrashNet 数据集（课程样例图片 `metal27.jpg` 等即来自 TrashNet）。删除标签冲突的重复图片，把同一物体的近重复照片归为一组后按组分层划分 70 / 15 / 15，同一物体不会同时出现在训练集和测试集。标注统一转成 COCO 格式，数据集类仍然按 bbox 裁剪，和课程任务一致。
- **数据加载**：所有图片统一 `convert("RGB")`，bbox 裁剪到图像边界，类别 id 自动映射到 `0..K-1`，类别数从标注文件读取。
- **训练**：修复最佳模型保存与 train/eval 模式；按验证集 macro-F1 选模型并早停，测试集只在最后评估一次；数据增强（RandomResizedCrop、翻转、颜色扰动）、ImageNet 均值方差归一化、AdamW（BN 和偏置不加 weight decay）+ label smoothing + warmup/cosine 学习率；多随机种子；支持 CUDA / Apple MPS / CPU。在 Linux 上把裁剪后的图片预先缩放并缓存在内存里，减轻 Colab 只有 2 个 CPU 时的数据加载瓶颈。
- **模型**：同一个 `AlexNet` 类支持原版、加 BatchNorm、加载 ImageNet 预训练权重三种用法（参数名与 torchvision 一致）。
- **评估**：统一混淆矩阵方向（行=真实，列=预测），输出准确率、macro-F1、每类 precision / recall / F1，保存混淆矩阵和训练曲线图；Grad-CAM 可视化模型关注的区域；`predict.py` 可直接对任意图片分类。
- **基线**：改成 sklearn Pipeline，PCA 只在训练集上拟合并保留 95% 方差；在验证集上调参；新增 RBF-SVM 对比。
- **工程**：`src/` 包 + 命令行脚本 + `pytest` 单元测试和端到端测试 + GitHub Actions CI。

## 项目结构

```
.
├── src/wastecls/
│   ├── data.py              # CocoCropDataset（按 bbox 裁剪）、数据增强、分层划分
│   ├── models.py            # AlexNet / AlexNet-BN / ImageNet 预训练 AlexNet
│   ├── engine.py            # 训练与推理循环
│   ├── metrics.py           # 混淆矩阵、macro-F1、每类指标、绘图
│   ├── dedup.py             # 标签冲突检测、近重复图片分组（并查集）
│   ├── gradcam.py           # Grad-CAM
│   ├── baseline.py          # HOG + 标准化 + PCA + 分类器 Pipeline
│   └── utils.py             # 随机种子、设备选择
├── scripts/
│   ├── prepare_trashnet.py  # TrashNet zip → 去重、分组 → COCO 标注 + train/val/test 划分
│   ├── train.py             # 训练 AlexNet（--preset course / scratch / scratch_bn / finetune，--seed）
│   ├── evaluate.py          # 评估已保存的 checkpoint
│   ├── predict.py           # 对任意图片分类，可输出 Grad-CAM
│   ├── gradcam_grid.py      # 测试集上每类最有把握的正确 / 错误预测的 Grad-CAM
│   ├── train_baseline.py    # HOG + 决策树 / SVM 基线
│   └── summarize.py         # 汇总各实验（多种子取均值 ± 标准差）为表格
├── notebooks/
│   └── colab_run_all.ipynb  # Colab 一键运行全部实验（结果可存到 Google Drive，断线可续跑）
├── tests/                   # 单元测试 + 端到端测试
├── .github/workflows/       # CI：每次提交自动运行测试
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

**Colab（推荐）**：点击上方 *Open in Colab*，切换到 GPU 运行时后全部运行（T4 上约 1.5–2 小时）。

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
python scripts/train_baseline.py                         # HOG + 决策树 / SVM
python scripts/train.py --preset course --seed 0         # 原课程配方 -> results/course/seed0/
python scripts/train.py --preset scratch --seed 0        # AlexNet 从零训练，改进配方
python scripts/train.py --preset scratch_bn --seed 0     # AlexNet-BN 从零训练
python scripts/train.py --preset finetune --seed 0       # ImageNet 预训练 AlexNet 微调
python scripts/summarize.py                              # 生成 results/summary.md
```

命令行参数可以覆盖 preset 中的任意设置，例如 `--epochs 50 --lr 3e-4`。

```bash
python scripts/predict.py --checkpoint results/finetune/seed0/best.pt photo.jpg --gradcam cam.png
python scripts/gradcam_grid.py --checkpoint results/finetune/seed0/best.pt
```

## 局限

- TrashNet 没有物体编号，"同一物体"是用特征相似度推断的。相似度低于 0.85 的同一物体照片（例如角度差别很大时）仍可能分到不同划分，测试结果可能略微偏高。
- 每张图只有一个物体、背景单一，模型在真实场景（杂乱背景、多物体）中的效果需要另行验证。
- 测试集中 trash 类只有 21 张，这一类的指标波动较大。

实验报告和课程讲义未包含在仓库中。
