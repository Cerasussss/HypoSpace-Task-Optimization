from modelscope import snapshot_download
import os

# 设置模型存储路径
model_path = "/opt/data/private/HypoSpace-40DA/models"
os.makedirs(model_path, exist_ok=True)

# 下载 Qwen3 模型
model_dir = snapshot_download(
    'qwen/Qwen2-7B-Instruct',
    cache_dir=model_path
)

print(f"========Model successfully download to: {model_dir}========")