# 使用本地Qwen模型进行因果评估

本文档说明如何使用本地下载的Qwen模型进行因果评估，而不是通过API调用远程模型。

## 依赖安装

首先确保安装了必要的依赖：

```bash
pip install transformers torch
```

## 下载Qwen模型

您可以使用huggingface-cli下载Qwen模型（以Qwen2-7B为例）：

```bash
# 使用huggingface-cli下载模型
huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir /path/to/your/qwen/model
```

或者在Python中直接加载（会自动下载）：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

## 配置文件

使用本地Qwen模型需要创建一个配置文件。示例配置文件 `config/config_qwen_local.yaml`：

```yaml
llm:
  type: qwen_local
  models:
    qwen_local: "/path/to/your/qwen/model"  # 修改为你的本地模型路径
  temperature: 0.7
  max_tokens: 512

benchmark:
  checkpoint: "checkpoints"
  verbose: true
  output_pattern: "results/{dataset_name}_{model}.json"
```

## 运行评估

使用以下命令运行评估：

```bash
cd /opt/data/private/HypoSpace-40DA/causal
python run_causal_benchmark.py \
  --dataset "../datasets/node03/n3_all_observations.json" \
  --config "config/config_qwen_local.yaml" \
  --n-samples 30 \
  --query-multiplier 1.0 \
  --seed 33550336
```

## 优势

使用本地Qwen模型的优势：
1. 无需网络连接
2. 无API调用费用
3. 更快的响应速度（取决于本地硬件）
4. 更好的隐私保护

## 硬件要求

运行Qwen2-7B模型的推荐硬件：
- GPU: 至少8GB显存（推荐16GB或更多）
- CPU: 现代多核处理器
- 内存: 至少16GB RAM（推荐32GB或更多）