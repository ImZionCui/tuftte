# 使用部分数据测试说明

## 修改内容

已修改代码以支持使用部分 .hist 文件进行快速测试，并添加了文件名后缀以区分不同模式：

### 1. 核心修改：`utils/NetworkParser.py`

在 `parse_histories()` 函数中添加了两个新参数：
- `num_train_files`: 指定使用多少个训练 .hist 文件（默认 None = 使用全部18个）
- `num_test_files`: 指定使用多少个测试 .hist 文件（默认 None = 使用全部6个）

### 2. 核心修改：`algorithms/TUFTTESolver.py`

在 `TUFTTESolver` 中添加了 `suffix` 参数：
- `suffix`: 文件名后缀（默认 "" = 无后缀，可指定 "_quick" 等标识符）
- 自动应用于所有生成的模型文件和目录名

例如：
- `model_parameter{suffix}.pkl` （如 `model_parameter_quick.pkl`）
- `model_{type}{suffix}.pkl` （如 `model_A_quick.pkl`）
- `opts_{type}{suffix}/` （如 `opts_A_quick/`）

### 3. 新测试脚本和更新

#### `test_partial_data.py`（新创建）
专门用于测试部分数据的脚本，配置为：
- **训练集**: 6个 .hist 文件（1-6.hist，约12,096个TM，6周数据）
- **测试集**: 2个 .hist 文件（19-20.hist，约4,032个TM，2周数据）
- **文件后缀**: `_quick` （生成 `model_*_quick.pkl` 等）

运行方式：
```bash
python test_partial_data.py
```

#### `test_parameter_availability.py`（已更新）
已更新 `setup_network()` 函数，支持：
- `num_train_files`: 限制训练文件数
- `num_test_files`: 限制测试文件数
- `num_train_samples`: 进一步限制训练样本数（在文件加载后）
- `num_test_samples`: 限制测试样本数

**注意**：此脚本使用原始的 TUFTTESolver（无后缀），生成的文件为 `model_A.pkl` 等

#### `test_param_quick.py`（已更新）
更新为：
- 使用6个训练文件和2个测试文件
- 减少到1000训练样本和10测试样本
- **使用 `_quick` 后缀**，生成 `model_*_quick.pkl` 等文件

## 数据量对比

### 完整数据（原始）
- 训练：18个 .hist 文件 = 18周 × 2016 TM/周 = **36,288 TM** ≈ 4.5个月
- 测试：6个 .hist 文件 = 6周 × 2016 TM/周 = **12,096 TM** ≈ 1.5个月
- 总计：6个月数据

### 部分数据（测试用）
- 训练：6个 .hist 文件 = 6周 × 2016 TM/周 = **12,096 TM** ≈ 1.5个月
- 测试：2个 .hist 文件 = 2周 × 2016 TM/周 = **4,032 TM** ≈ 0.5个月
- 总计：2个月数据

### 快速测试（极小数据）
- 训练：1000 TM（从6个文件中抽取）
- 测试：10 TM（从2个文件中抽取）

## 推荐测试流程

1. **第一步：快速验证**（1-2分钟）
   ```bash
   python test_param_quick.py
   ```
   使用最小数据量验证代码是否能正常运行

2. **第二步：部分数据测试**（10-30分钟）
   ```bash
   python test_partial_data.py
   ```
   使用6+2文件验证训练流程和结果质量

3. **第三步：完整数据训练**（如果第二步结果良好）
   修改脚本移除文件数限制，使用全部18+6文件进行完整训练

## 注意事项

1. **兼容性**：未指定 `num_train_files`/`num_test_files` 时，行为与原始代码完全相同
2. **文件顺序**：使用前N个文件（按文件名排序），保持时间连续性
3. **文件后缀**：
   - 使用 `suffix="_quick"` 的脚本生成 `model_*_quick.pkl` 等带后缀的文件
   - 不指定或使用 `suffix=""` 的脚本生成 `model_*.pkl` 等无后缀文件
   - 这样可以同时保存不同模式的结果，避免混淆和误删

4. **时间估算**：
   - 1000 TM训练：约1-2分钟
   - 12,096 TM训练（6文件）：约10-30分钟
   - 36,288 TM训练（18文件）：约30-90分钟
   （具体时间取决于硬件配置）

5. **模型保存**：部分数据训练的模型会保存到：
   - `data/Abilene/model_parameter_quick.pkl`
   - `data/Abilene/model_A_quick.pkl` 或 `model_D_quick.pkl`
   
   完整训练前建议备份或重命名这些文件

## 使用示例

### 示例1：快速测试（带_quick后缀）
```python
from utils.NetworkParser import parse_topology, parse_histories
from algorithms.TUFTTESolver import TUFTTESolver, Availability

network = parse_topology("Abilene")
parse_histories(network, num_train_files=6, num_test_files=2)
# ... 其他设置 ...
solver = TUFTTESolver(network, hist_len=12, type=Availability, suffix="_quick")
solver.solve()
# 生成: model_parameter_quick.pkl, model_A_quick.pkl, opts_A_quick/...
```

### 示例2：完整数据（无后缀）
```python
network = parse_topology("Abilene")
parse_histories(network)  # 使用全部18+6文件
# ... 其他设置 ...
solver = TUFTTESolver(network, hist_len=12, type=Availability)  # 无suffix参数
solver.solve()
# 生成: model_parameter.pkl, model_A.pkl, opts_A/...
```

### 示例3：自定义后缀
```python
# 可以使用任意后缀标识
solver = TUFTTESolver(network, hist_len=12, type=Availability, suffix="_v2")
solver.solve()
# 生成: model_parameter_v2.pkl, model_A_v2.pkl, opts_A_v2/...
```

### 示例4：组合使用
```python
# 先限制文件数，再限制样本数
parse_histories(network, num_train_files=6, num_test_files=2)
network.reduce_data(1000, 10)  # 进一步减少到1000+10样本
solver = TUFTTESolver(network, suffix="_minimal")  # 自定义后缀
solver.solve()
```
