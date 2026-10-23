# 任务 B：行程时间估计代码说明

本目录为任务 B 的代码。方法采用：

```text
预测时间 = 点数中位数 baseline + Transformer 预测残差
```

其中，点数中位数 baseline 利用约 15 秒降采样带来的时间先验；Transformer 进一步学习路径形状、距离、起终点位置、停滞比例和出发时间等因素带来的残差。

## 文件说明

```text
task_b_common.py        公共函数、特征提取、baseline、Dataset、模型结构和评估指标
train_task_b.py         训练 Residual Transformer，并保存 .pt 模型文件
predict_task_b.py       读取输入 pkl 和训练好的模型，生成提交格式的预测 pkl
plot_report_figures.py  生成报告中使用的可视化图
README.md              本说明文件
```

## 数据路径要求

默认假设在项目根目录下存在：

```text
student_release/data_ds15/train.pkl
student_release/task_B_tte/val_input.pkl
student_release/task_B_tte/val_gt.pkl
```

课堂测试时，老师会提供：

```text
student_release/task_B_tte/test_input.pkl
```

## 环境

需要 Python、NumPy 和 PyTorch。若使用 GPU，请保证 PyTorch 能够正常调用 CUDA。

本实验使用环境示例：

```text
Python 3.12
PyTorch 2.7.1
CUDA 可用
```

## 训练模型

在项目根目录运行，例如：

```bash
python code/task_b/train_task_b.py \
  --data-root student_release \
  --epochs 30 \
  --batch-size 384 \
  --d-model 64 \
  --layers 2 \
  --heads 4 \
  --patience 10 \
  --min-epochs 20 \
  --out code/task_b/artifacts/task_b_transformer.pt
```

训练脚本会先打印点数中位数 baseline 的验证集指标，然后逐轮输出 Transformer 的验证集 MAE、RMSE 和 MAPE。训练完成后，模型权重和标准化参数会保存在 `--out` 指定的 `.pt` 文件中。

## 验证集预测

训练完成后，可以在验证集上生成预测并计算指标：

```bash
python code/task_b/predict_task_b.py \
  --input student_release/task_B_tte/val_input.pkl \
  --gt student_release/task_B_tte/val_gt.pkl \
  --model code/task_b/artifacts/task_b_transformer.pt \
  --output code/task_b/val_pred.pkl \
  --no-progress
```

输出文件格式为：

```python
[
    {"traj_id": 0, "travel_time": 1024},
    {"traj_id": 1, "travel_time": 1387}
]
```

## 课堂测试预测

将老师提供的 `test_input.pkl` 放到任务 B 目录后，运行：

```bash
python code/task_b/predict_task_b.py \
  --input student_release/task_B_tte/test_input.pkl \
  --model code/task_b/artifacts/task_b_transformer.pt \
  --output code/task_b/test_pred.pkl \
  --no-progress
```

最终提交：

```text
code/task_b/test_pred.pkl
```

## 仅运行 baseline

如果不使用 Transformer，也可以直接运行点数中位数 baseline：

```bash
python code/task_b/predict_task_b.py \
  --baseline-only \
  --input student_release/task_B_tte/val_input.pkl \
  --gt student_release/task_B_tte/val_gt.pkl \
  --train-data student_release/data_ds15/train.pkl \
  --output code/task_b/val_pred_baseline.pkl \
  --no-progress
```

baseline 不需要 `.pt` 模型文件，但需要训练集 `data_ds15/train.pkl` 来统计不同点数下的行程时间中位数。

