# test_param_quick.py 运行后生成的文件

## 文件生成总结

运行 `python test_param_quick.py` 后会生成以下文件（带 `_quick` 后缀以区分其他模式）：

## 1. 模型文件（核心输出）

### a) 参数配置网络模型
```
data/Abilene/model_parameter_quick.pkl  (~5-10 MB)
```
- **用途**：参数神经网络，输出theta参数（约束缩放系数）
- **生成时机**：首次运行时自动生成，之后直接加载
- **内容**：ParameterNN模型权重和结构
- **创建条件**：如果文件不存在则训练20个epoch
- **后缀说明**：`_quick` 表示来自快速测试模式

### b) 完整TUFTTE模型
```
data/Abilene/model_A_quick.pkl  (~20-30 MB)  # 用于Availability类型
或
data/Abilene/model_D_quick.pkl  (~20-30 MB)  # 用于DemandLoss类型
```
- **用途**：包含参数网络和优化层的完整模型
- **生成时机**：首次运行时训练，之后直接加载
- **内容**：
  - 参数网络（ParameterNN）
  - CvxpyLayer（差分优化层）
  - 网络约束信息
- **创建条件**：如果文件不存在则训练
- **后缀说明**：`_quick` 表示来自快速测试模式

## 2. 最优值文件

### 优化目标值缓存
```
data/Abilene/opts_A_quick/int_1.0.opt  # 对应Availability
或
data/Abilene/opts_D_quick/int_1.0.opt  # 对应DemandLoss
```
- **用途**：训练数据的最优损失值
- **大小**：约 1-2 MB（一行一个损失值）
- **内容**：每个训练时间步的最优值（浮点数，每行一个）
- **行数**：约 12,000+ 行（从第12个TM开始，跳过初始hist_len=12）
- **生成时机**：首次运行时计算（耗时较长），之后复用
- **用途**：训练时作为监督信号
- **后缀说明**：`opts_A_quick` 或 `opts_D_quick` 表示来自快速测试模式

## 3. 完整的目录结构

运行后的目录结构如下：

```
data/Abilene/
├── Abilene_int.txt          （网络拓扑信息）
├── Abilene_int.pickle.nnet  （路由矩阵）
├── tunnels.txt              （隧道定义）
├── train/                   （训练.hist文件）
│   ├── 1.hist
│   ├── 2.hist
│   └── ...
├── test/                    （测试.hist文件）
│   ├── 19.hist
│   └── 20.hist
├── opts_A_quick/            （新生成的目录 - quick模式）
│   └── int_1.0.opt          （最优值文件）
├── opts_A/                  （其他模式生成的目录）
│   └── int_1.0.opt
├── model_parameter_quick.pkl    （新生成 - 来自test_param_quick.py）
├── model_parameter.pkl          （来自其他脚本）
├── model_A_quick.pkl            （新生成 - 来自test_param_quick.py）
├── model_A.pkl                  （来自其他脚本）
└── ...
```

## 4. 文件生成流程

### 第一次运行 test_param_quick.py：

```
开始运行
  ↓
检查 model_parameter.pkl 是否存在？
  ├→ 否：训练参数网络 → 保存 model_parameter.pkl (耗时 20-30秒)
  └→ 是：直接加载
  ↓
检查 model_A.pkl 是否存在？
  ├→ 否：
  │   ├ 检查 opts_A/int_1.0.opt 是否存在？
  │   │  ├→ 否：计算最优值 → 保存 opts_A/int_1.0.opt (耗时 5-10分钟)
  │   │  └→ 是：加载
  │   └ 训练完整模型 → 保存 model_A.pkl (耗时 10-20秒)
  └→ 是：直接加载
  ↓
在测试数据上运行求解
  ↓
完成
```

### 第二次运行 test_param_quick.py：

```
开始运行
  ↓
加载 model_parameter.pkl
  ↓
加载 model_A.pkl
  ↓
加载 opts_A/int_1.0.opt
  ↓
在测试数据上运行求解
  ↓
完成 (快速，只需1-2秒)
```

## 5. 文件内容说明

### model_parameter_quick.pkl 内部结构
```python
ParameterNN(
  fc1: Linear(in_features=132, out_features=256)
  relu1: ReLU()
  fc2: Linear(in_features=256, out_features=128)
  relu2: ReLU()
  fc3: Linear(in_features=128, out_features=132)  # 输出theta参数
  sigmoid: Sigmoid()  # 限制到[0.5, 1.5]范围
)
```

### model_A_quick.pkl 内部结构
```python
TEAVARModel(
  param_nn: ParameterNN       # 从model_parameter_quick.pkl加载
  cvxpy_layer: CvxpyLayer     # 差分LP优化层
  network_info: {...}         # 网络拓扑信息
)
```

### opts_A_quick/int_1.0.opt 格式
```
损失值1
损失值2
损失值3
...
损失值N
```
每一行是一个浮点数，代表对应时间步的最优损失值。

## 6. 文件大小估算

| 文件 | 大小 | 说明 |
|------|------|------|
| model_parameter_quick.pkl | 5-10 MB | 参数网络模型 |
| model_A_quick.pkl | 20-30 MB | 完整优化模型 |
| opts_A_quick/int_1.0.opt | 1-2 MB | 最优值缓存 |
| 总计 | ~30-50 MB | 所有quick模式文件 |

## 7. 运行时间估算

**第一次运行**（生成所有文件）：
- model_parameter.pkl 训练：~20-30秒
- opts 计算：~5-10分钟（**最耗时**）
- model_A.pkl 训练：~10-20秒
- 总计：**~6-11分钟**

**第二次及以后运行**（所有文件已存在）：
- 直接加载模型：~1-2秒
- 推理：~1-2秒
- 总计：**~2-4秒**

## 8. 模型持久化逻辑

```python
# 伪代码
def _pre_train():
    model_path = f"data/{name}/model_parameter.pkl"
    if not exists(model_path):
        # 训练参数网络
        model = ParameterNN(...)
        train_model(model)
        torch.save(model, model_path)  # 保存
    return torch.load(model_path)      # 加载

def _train():
    model_path = f"data/{name}/model_{type}.pkl"
    if not exists(model_path):
        # 计算最优值
        opts = _compute_opts_to_train()  # 生成.opt文件
        # 训练完整模型
        model = TEAVARModel(param_model, network)
        train_model(model)
        torch.save(model, model_path)  # 保存
    return torch.load(model_path)      # 加载
```

## 9. 清理文件

如果要重新训练（忽略缓存），可以删除 `_quick` 后缀的文件：

```bash
# 删除quick模式的模型（会重新训练）
rm data/Abilene/model_parameter_quick.pkl
rm data/Abilene/model_A_quick.pkl

# 删除quick模式的最优值缓存（会重新计算）
rm data/Abilene/opts_A_quick/int_1.0.opt

# 删除所有quick模式缓存
rm -rf data/Abilene/model_*_quick.pkl
rm -rf data/Abilene/opts_*_quick/
```

注意：删除 `_quick` 后缀的文件不会影响其他模式的文件。

## 10. 与其他模式的文件区分

| 来源 | model_parameter | model_A | opts_A | 备注 |
|------|-----------------|---------|--------|------|
| test_param_quick.py | model_parameter_quick.pkl | model_A_quick.pkl | opts_A_quick/ | **_quick后缀** |
| test_partial_data.py | model_parameter_quick.pkl | model_A_quick.pkl | opts_A_quick/ | **_quick后缀** |
| test_parameter_availability.py | model_parameter.pkl | model_A.pkl | opts_A/ | 无后缀 |
| main.py 或其他脚本 | model_parameter.pkl | model_A.pkl | opts_A/ | 无后缀 |

这样可以轻松区分：
- `_quick` 后缀 = 快速测试模式（使用部分数据）
- 无后缀 = 完整数据模式

## 11. 重要提示

⚠️ **第一次运行会比较慢！** 特别是计算最优值部分，可能需要5-10分钟。

这是因为：
1. 需要为每个训练TM运行Gurobi求解器（~12,000次）
2. Gurobi调用开销较大
3. opts文件很大（12,000+ 行）

但这个计算只进行一次，之后直接从文件加载，速度会快得多。

💡 **后缀的作用**：
- 同时保存不同模式的结果，便于对比
- 避免误删重要文件
- 支持快速和完整模式的并行测试
