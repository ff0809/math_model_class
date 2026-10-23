# 任务 A：轨迹缺失点恢复代码说明

本目录为任务 A 的代码，包含三个部分：

```text
baseline/     线性插值 baseline
astar/        路网 A* 对比实验原型
retrieval/    历史轨迹模板检索方法，最终采用
transformer/  Transformer 与路网特征对比实验
```

最终方法采用历史轨迹模板检索：先把经纬度投影到局部平面坐标，再按缺失段长度建立历史轨迹片段索引；预测时根据缺失段两端点和上下文检索相似历史片段，把历史片段相对线性插值的残差加回当前轨迹。

## 文件说明

```text
baseline/linear.py                  线性插值 baseline
baseline/result_check.py            baseline 结果检查
baseline/plot_*.py                  baseline 可视化脚本
baseline/compare_trajectories.py    预测轨迹与真实轨迹对比图

astar/route_recover.py              路网 A* 约束恢复原型

retrieval/retrieval_recover.py      最终方法主程序
retrieval/tune_retrieval.py         参数搜索脚本
retrieval/inspect_results.py        结果检查脚本
retrieval/plot_report_figures.py    报告图片生成脚本

transformer/*.py                    Transformer 对比实验代码
```

其中真正用于生成任务 A 提交结果的是 `retrieval/retrieval_recover.py`。`astar/` 和 `transformer/` 用于对比实验，画图脚本和 `inspect_results.py` 只用于实验分析，不影响最终预测。

## 数据路径要求

默认假设在项目根目录下存在：

```text
student_release/data_ds15/train.pkl
student_release/task_A_recovery/val_input_8.pkl
student_release/task_A_recovery/val_input_16.pkl
student_release/task_A_recovery/val_gt.pkl
```

新输入数据只需要保持与 `val_input_8.pkl` 或 `val_input_16.pkl` 相同格式即可。

## 环境

历史模板检索方法需要：

```text
Python
NumPy
SciPy
pyproj
```

若运行 Transformer 对比实验，还需要 PyTorch。最终检索方法不依赖 GPU。

若运行路网 A* 对比实验，还需要 GeoPandas、NetworkX 和 Shapely，并准备 OSM 路网 Shapefile。

## 最终方法预测

在项目根目录运行，例如对 1/8 采样验证集预测：

```bash
python code/task_a/retrieval/retrieval_recover.py \
  --input student_release/task_A_recovery/val_input_8.pkl \
  --gt student_release/task_A_recovery/val_gt.pkl \
  --output code/task_a/outputs/val_input_8_retrieval.pkl \
  --cache code/task_a/artifacts/retrieval_cache.pkl \
  --use-context
```

对 1/16 采样验证集预测：

```bash
python code/task_a/retrieval/retrieval_recover.py \
  --input student_release/task_A_recovery/val_input_16.pkl \
  --gt student_release/task_A_recovery/val_gt.pkl \
  --output code/task_a/outputs/val_input_16_retrieval.pkl \
  --cache code/task_a/artifacts/retrieval_cache.pkl \
  --use-context
```

输出文件格式为：

```python
[
    {"traj_id": 0, "coords": [[lon, lat], [lon, lat], ...]},
    {"traj_id": 1, "coords": [[lon, lat], [lon, lat], ...]}
]
```

第一次运行会从 `train.pkl` 建立模板索引，并保存到 `--cache` 指定位置；之后再次运行会直接读取缓存。若训练集或参数发生变化，可加入 `--rebuild-cache` 重新构建缓存。

## 新数据预测

拿到新的输入文件后，将 `--input` 改成对应路径，并去掉 `--gt`：

```bash
python code/task_a/retrieval/retrieval_recover.py \
  --input student_release/task_A_recovery/test_input.pkl \
  --output code/task_a/outputs/test_pred.pkl \
  --cache code/task_a/artifacts/retrieval_cache.pkl \
  --use-context
```

最终提交 `--output` 生成的 `.pkl` 文件。

## 运行线性插值 baseline

`baseline/linear.py` 是原始线性插值基线脚本，默认读取 `student_release/task_A_recovery/val_input_16.pkl` 并输出 `submission_val_input_16.pkl`：

```bash
python code/task_a/baseline/linear.py
```

该方法只根据缺失段两端点做直线插值，用于和最终检索方法对比。

## Transformer 对比实验

Transformer 代码保留为对比实验，入口为：

```bash
python code/task_a/transformer/train.py
python code/task_a/transformer/eval.py
python code/task_a/transformer/inference.py
```

该部分用于验证深度学习和路网特征方案的效果，不是最终采用的主方法。

## 路网 A* 对比实验

路网 A* 原型入口为：

```bash
python code/task_a/astar/route_recover.py \
  --input student_release/task_A_recovery/val_input_8.pkl \
  --gt student_release/task_A_recovery/val_gt.pkl \
  --road-shp task_a/shaanxi-260507-free.shp/gis_osm_roads_free_1.shp \
  --output code/task_a/outputs/val_input_8_astar.pkl \
  --limit 100
```

该方法用于检验静态路网最短路约束的效果，报告中作为对比实验展示。
