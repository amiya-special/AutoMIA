# AutoMia LLaVA Agent 快速运行说明



## 🎞️ VL-MIA Datasets

The **VL-MIA datasets** serve as a benchmark designed to evaluate membership inference attack (MIA) methods for large vision language models. Access our **VL-MIA datasets** directly on [image](https://huggingface.co/datasets/JaineLi/VL-MIA-image) and [text](https://huggingface.co/datasets/JaineLi/VL-MIA-text) . 

#### Loading the Datasets

```python
from datasets import load_dataset

text_len = 64 # 16,32,64

img_subset = "img_Flickr" # or img_dalle
text_subset = "llava_v15_gpt_text" # or minigpt4_stage2_text

image_dataset = load_dataset("JaineLi/VL-MIA-image", subset, split='train')
text_dataset = load_dataset("JaineLi/VL-MIA-text", subset, split=f"length_{text_len}")
```




本文档用于快速运行以下两个脚本：

- `run_with_text_agent.py`（文本 MIA 流程）
- `run_with_img_agent.py`（图像 MIA 流程）

## 0. 环境与可编辑安装（pip install -e .）

若希望像 VL-MIA-main 一样使用 `pip install -e .` 安装依赖并注册包，请按下面步骤操作。

### 0.1 导出当前 base 环境依赖（可选）

在**已激活的** base（或当前使用的 conda/venv）环境中执行，将当前环境的所有包版本导出到文件，便于复现或生成 `requirements.txt`：

```bash
# 方式一：pip 导出（推荐，得到 pip 可安装的格式）
pip list --format=freeze > requirements.txt

# 方式二：更紧凑，仅包名+版本
pip freeze > requirements-freeze.txt
```

之后可在新环境中用 `pip install -r requirements.txt` 安装。若你已有可运行环境，也可以不导出，直接使用下面的可编辑安装。

### 0.2 在 AutoMia 根目录做可编辑安装

依赖定义在 **AutoMia 根目录** 的 `pyproject.toml` 中，安装时需在 **AutoMia 根目录** 执行（不是 `llava` 目录）：

```bash
cd /root/autodl-tmp/AutoMia
pip install --upgrade pip   # 建议，以支持 PEP 660 等
pip install -e .
```

安装完成后：

- `agent_mia` 会作为包被安装，在任意目录都可 `import agent_mia`。
- 运行脚本仍需在 **llava 目录** 下执行（见下文），以便正确找到 `eval`、`metric_util`、`llama` 等模块（它们需在 llava 同目录或由 `sys.path` 提供）。

### 0.3 使用导出的 requirements 安装（可选）

若已用 0.1 导出 `requirements.txt`，可在新环境中先安装再可编辑安装：

```bash
pip install -r requirements.txt
cd /root/autodl-tmp/AutoMia
pip install -e .
```

### 0.4 使用 setup.py 安装（基于 llava/requirements.txt）

本目录下提供 `setup.py`，会读取同目录的 `requirements.txt` 作为依赖列表进行安装。在 **llava 目录** 下执行：

```bash
cd /root/autodl-tmp/AutoMia/llava
pip install --upgrade pip
pip install -e .
```

或仅安装依赖（不注册为可编辑包）：

```bash
cd /root/autodl-tmp/AutoMia/llava
pip install -r requirements.txt
```

> 说明：`setup.py` 会跳过 `pip`、`conda`、`llava==...` 等项，并将 `torch==2.1.2+cu118` 转为 `torch==2.1.2` 以便从 PyPI 安装；若需 CUDA 版 PyTorch，请先单独安装后再执行上述命令。

---

## 1. 运行前准备

### 1.1 进入目录

运行两个 run 脚本时，请进入 **llava** 目录：

```bash
cd /root/autodl-tmp/AutoMia/llava
```

### 1.2 Python 环境与依赖

若已按 **0.2** 在 AutoMia 根目录执行过 `pip install -e .`，则主要依赖已由 `pyproject.toml` 安装。否则请确保环境中已安装至少：

- `torch`、`torchvision`
- `datasets`
- `numpy`、`scipy`
- `tqdm`
- `Pillow`
- `requests`
- `opencv-python`
- `openai`
- `PyYAML`

> 说明：脚本还会依赖项目内模块（如 `llama`、`agent_mia`、`metric_util`、`eval` 等）。请在项目根目录结构完整、且从 **llava** 目录运行的前提下使用。

### 1.3 模型权重

两个脚本都支持：

- 显式传入 `--llama_path /your/llama_weights_dir`
- 或不传，自动使用默认缓存目录：`~/.cache/vl-mia/llama_model_weights`

checkpoint 下载目录可通过环境变量覆盖：

```bash
export VL_MIA_CKPT_DIR=/your/checkpoints_dir
```

## 2. 快速运行：文本脚本

脚本：`run_with_text_agent.py`

### 2.1 最短可跑命令

```bash
python run_with_text_agent.py \
  --gpu_id 0 \
  --dataset_path /your/text_dataset_hf_disk \
  --output_dir ./outputs_text
```

### 2.2 常用参数

- `--dataset_path`：文本数据集路径（`datasets.load_from_disk` 格式）
- `--text_len`：文本长度（默认 `32`；默认数据路径依赖它）
- `--rounds`：迭代轮数（默认 `10`）
- `--n_metrics`：每轮生成的新指标数量（默认 `5`）
- `--model_type`：默认 `LORA-BIAS-7B-v21`
- `--llama_path`：LLaMA 权重路径（可选）

### 2.3 默认数据路径逻辑

如果不传 `--dataset_path`，脚本会尝试读取：

- `./VL-MIA-text_{text_len}`（相对 `run_with_text_agent.py` 所在目录）

## 3. 快速运行：图像脚本

脚本：`run_with_img_agent.py`

### 3.1 最短可跑命令

```bash
python run_with_img_agent.py \
  --gpu_id 0 \
  --local_dataset_path /your/image_dataset_hf_disk \
  --output_dir ./outputs_img
```

### 3.2 常用参数

- `--local_dataset_path`：图像数据集路径（`datasets.load_from_disk` 格式）
- `--instruction_prompt`：图像描述提示词（默认 `"Describe this image concisely."`）
- `--num_gen_token`：生成 token 数（默认 `32`）
- `--rounds`：迭代轮数（默认 `3`）
- `--model_type`：默认 `LORA-BIAS-7B-v21`
- `--llama_path`：LLaMA 权重路径（可选）

### 3.3 环境变量方式指定图像数据集

可替代 `--local_dataset_path`：

```bash
export MIA_IMAGE_DATASET_DIR=/your/image_dataset_hf_disk
python run_with_img_agent.py --gpu_id 0
```

## 4. 输出结果在哪里

两类输出目录来源不同，请注意：

- 实验 AUC 结果：由 `--output_dir` 控制
  - 文本：`{output_dir}/round_{k}/length_{text_len}/auc.txt`
  - 图像：`{output_dir}/{dataset}/gen_{num_gen_token}_tokens/round_{k}/...`
- Agent 轮次中间产物（指标提案、评估 JSON、策略积累）：
  - 固定写到 `./agent_round_outputs/`
  - 策略库默认在：
    - 文本：`./agent_mia/agent/strategy_bank_text.json`
    - 图像：`./agent_mia/agent/strategy_bank_img.json`

## 5. 数据集格式建议（最小要求）

两个脚本都用 `datasets.load_from_disk(...)` 读取数据，因此请先将数据集保存为 HuggingFace Dataset 磁盘格式。

- 文本脚本推理时会读取字段：`input`
- 图像脚本推理时会读取字段：`image`

另外，评估函数通常还需要成员标签等字段（由项目内 `eval/metric` 逻辑消费）。若你使用仓库已有数据处理流程导出的数据集，一般可直接运行。

## 6. 常见问题

- 报错 `Dataset path is not set`（图像）  
  说明未提供 `--local_dataset_path`，也未设置 `MIA_IMAGE_DATASET_DIR`。

- 报错找不到 LLaMA 权重  
  传入正确的 `--llama_path`，或确认默认缓存目录下权重完整。

- 只想先跑通流程  
  建议先用 `--rounds 1`，确认输出路径与环境无误后再加轮数。
