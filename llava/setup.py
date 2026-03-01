# -*- coding: utf-8 -*-
"""AutoMia llava 运行环境安装：根据 requirements.txt 安装依赖。

安装命令行（在 AutoMia/llava 目录下执行）：

  # 仅安装依赖（推荐）
  pip install -r requirements.txt

  # 或以可编辑方式安装当前项目（会读取 requirements.txt 作为 install_requires）
  pip install --upgrade pip
  pip install -e .

若已导出带 +cu118 的 torch，请先单独安装 CUDA 版 PyTorch，再执行上述命令。
"""
from setuptools import setup, find_packages
import os


def read_requirements(req_path="requirements.txt"):
    """从 requirements.txt 读取依赖，并做少量兼容处理。"""
    if not os.path.isfile(req_path):
        return []
    with open(req_path, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 跳过环境/工具类，避免与当前项目冲突
            if line.startswith("pip==") or line.startswith("conda=="):
                continue
            # 跳过 PyPI 上的 llava 包（与 VL-MIA/本地 LLaVA 冲突）
            if line.lower().startswith("llava=="):
                continue
            # PyPI 无 torch+cu118 等本地版本，改为不带 local 的版本
            if "torch==" in line and "+cu" in line:
                line = line.split("+")[0].strip()
            elif "torchvision==" in line and "+cu" in line:
                line = line.split("+")[0].strip()
            lines.append(line)
    return lines


install_requires = read_requirements(os.path.join(os.path.dirname(__file__), "requirements.txt"))

setup(
    name="automia-llava",
    version="0.1.0",
    description="AutoMia LLaVA Agent 运行依赖（run_with_text_agent / run_with_img_agent）",
    long_description=(
        open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8"
        ).read()
        if os.path.isfile(os.path.join(os.path.dirname(__file__), "README.md"))
        else ""
    ),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    install_requires=install_requires if install_requires else [
        "torch>=2.0",
        "torchvision",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "numpy",
        "scikit-learn",
        "scipy",
        "tqdm",
        "Pillow",
        "requests",
        "openai",
        "PyYAML",
    ],
    packages=[],  # 仅安装依赖，不声明包（脚本在本地直接运行）
)
