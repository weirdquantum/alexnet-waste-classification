# 基于 AlexNet 的垃圾图像分类 (AlexNet Waste Image Classification)

本科课程 **UESTC 3002: Artificial Intelligence & Machine Learning**（2024–2025 秋季学期）的 Lab Project。用 PyTorch 从零实现 AlexNet，在 COCO 格式标注的垃圾图像数据上完成"数据处理 → 网络搭建 → 训练 → 评估（准确率 + 混淆矩阵）"的完整流程；同时实现了一个传统机器学习基线（HOG 特征 + 标准化 + PCA + 决策树）作对比。

## 项目结构

```
.
├── README.md
└── Resources/
    ├── README.md              # 课程实验说明（英文，任务要求）
    ├── experiment.ipynb       # 主实验 notebook：数据 → 模型 → 训练 → 评估，含运行输出和混淆矩阵图
    │
    ├── dataset.py             # MyCOCODataset：按 bbox 裁剪目标 → 缩放到 192×192 → 返回 (HWC uint8 图像, 类别 id)
    ├── model.py               # AlexNet：5 层卷积特征提取 + AdaptiveAvgPool(6×6) + 3 层全连接分类头（Dropout 0.5）
    ├── train.py               # 训练：CrossEntropyLoss + Adam(lr=1e-3)，batch 32，按验证准确率保存 outputs/best_model.pth
    ├── evaluate.py            # 评估：加载权重，计算测试准确率与混淆矩阵
    │
    ├── mlmodel.py             # 传统 ML 基线：每个 RGB 通道提取 HOG → StandardScaler → PCA → DecisionTree，序列化为 outputs/model.pkl
    ├── ml_evaluate.py         # 基线评估：加载 model.pkl，输出混淆矩阵
    ├── outputs/
    │   └── model.pkl          # 训练好的 HOG+PCA+决策树基线模型
    │
    ├── data_coco/             # 样例数据（COCO 格式）
    │   ├── annotations.json   # 5 个类别：bottle / metal / glass / paper / cardboard
    │   └── JPEGImages/        # 每类 1 张示例图
    │
    ├── utils/
    │   ├── bbox_visualizer.py     # 基于 OpenCV 的检测框 / 标签绘制工具
    │   └── cocopytool_example.py  # pycocotools API 用法示例（读取 imgs / anns / cats）
    │
    └── 1/                     # 随报告提交的最终版代码（分模块整理）
        ├── Data_Processing.py     # COCOCustomDataset + torchvision transforms（Resize 224、ImageNet 均值方差归一化）
        ├── Network.py             # 7 分类 AlexNet（带逐层注释）
        ├── Model_Training.py      # 训练循环，CUDA 可用时自动用 GPU，保存最佳验证准确率权重
        └── Performance_Evaluate.py # classification_report + seaborn 混淆矩阵热力图
```

## 方法概述

| 环节 | 实现 |
|---|---|
| 数据处理 | 用 `pycocotools` 读取 COCO 标注，按每个实例的 bbox `[x, y, w, h]` 裁剪出目标，双线性插值缩放到固定尺寸，灰度图转 RGB；训练时像素除以 255 归一化并转为 `[B, C, H, W]` |
| 网络 | AlexNet：Conv(11×11, s4) → Conv(5×5) → 3 × Conv(3×3)，配合 ReLU 和 MaxPool；`AdaptiveAvgPool2d(6×6)` 使网络适配任意输入尺寸；分类头 9216 → 4096 → 4096 → num_classes |
| 训练 | `CrossEntropyLoss` + `Adam(lr=1e-3)`，batch size 32，每轮在验证集上计算准确率并保存最佳权重 |
| 评估 | 测试集准确率、混淆矩阵（`evaluate.py` 中手写，`1/Performance_Evaluate.py` 用 sklearn + seaborn 可视化） |
| 基线 | 3 通道 HOG 特征拼接 → `StandardScaler` → `PCA` → `DecisionTreeClassifier` |

## 运行

依赖：Python 3.x、PyTorch、torchvision、pycocotools、scikit-learn、scikit-image、matplotlib、seaborn、tqdm、opencv-python。

```bash
pip install torch torchvision pycocotools scikit-learn scikit-image matplotlib seaborn tqdm opencv-python
```

脚本中的数据路径写的是 `data/data_coco`，在 `Resources/` 下运行前先建一个软链接：

```bash
cd Resources && mkdir -p data && ln -s ../data_coco data/data_coco
```

```bash
python train.py          # 训练 AlexNet，权重保存到 outputs/best_model.pth
python evaluate.py       # 测试准确率 + 混淆矩阵
python mlmodel.py        # 训练 HOG + PCA + 决策树基线
python ml_evaluate.py    # 评估基线
```

也可以直接打开 `experiment.ipynb` 按顺序运行。

## 说明

- 仓库中的 `data_coco/` 只是每类 1 张图的样例，用于跑通流程；`experiment.ipynb` 里训练集、验证集、测试集都指向同一份标注，因此其中的 100% 测试准确率只说明流程能跑通，不代表模型的泛化能力。
- 课程要求的 AlexNet 输出为 7 类（`num_classes = 7`），样例标注只有 5 类，多出的两个输出不会被用到。
- `1/` 下是随报告提交的分模块代码，各文件按 notebook 的顺序依次执行（例如 `Model_Training.py` 依赖前面定义的 `AlexNet` 和 `train_loader`），数据路径需改为本地路径。
- 实验报告和课程讲义未包含在仓库中。
