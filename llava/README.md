
# AutoMia LLaVA Agent Quick Guide

This document provides a minimal setup and run guide for:

- `run_with_text_agent.py` (text MIA)
- `run_with_img_agent.py` (image MIA)

## 1) Install

Run editable install from the `AutoMia` root directory:

```bash
cd /root/autodl-tmp/AutoMia
pip install --upgrade pip
pip install -e .
```

## 2) Datasets

VL-MIA benchmark datasets are available on Hugging Face:

- Image: [JaineLi/VL-MIA-image](https://huggingface.co/datasets/JaineLi/VL-MIA-image)
- Text: [JaineLi/VL-MIA-text](https://huggingface.co/datasets/JaineLi/VL-MIA-text)

The run scripts use `datasets.load_from_disk(...)`, so prepare local datasets in Hugging Face disk format.

Minimum required fields:

- Text pipeline: `input`
- Image pipeline: `image`

## 3) Run

Go to the `llava` directory before running scripts:

```bash
cd /root/autodl-tmp/AutoMia/llava
```

Text:

```bash
python run_with_text_agent.py \
  --gpu_id 0 \
  --dataset_path /your/text_dataset_hf_disk \
  --rounds 10 \
  --model_type LORA-BIAS-7B-v21 \
  --llama_path /your/llama_weights_dir \
  --output_dir ./outputs_text
```

Image:

```bash
python run_with_img_agent.py \
  --gpu_id 0 \
  --local_dataset_path /your/image_dataset_hf_disk \
  --rounds 3 \
  --model_type LORA-BIAS-7B-v21 \
  --llama_path /your/llama_weights_dir \
  --output_dir ./outputs_img
```

## 4) Model Weights

You can provide `--llama_path /your/llama_weights_dir`, or use the default cache:

`~/.cache/vl-mia/llama_model_weights`

Optional checkpoint root override:

```bash
export VL_MIA_CKPT_DIR=/your/checkpoints_dir
```

## 5) Outputs

- Main AUC outputs are saved under `--output_dir`
- Agent intermediate files are saved under `./agent_round_outputs/`
- Strategy banks:
  - Text: `./agent_mia/agent/strategy_bank_text.json`
  - Image: `./agent_mia/agent/strategy_bank_img.json`

