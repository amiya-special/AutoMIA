<div align="center">
<h1><a href="https://arxiv.org/abs/2604.01014" target="_blank">AutoMIA: Improved Baselines for Membership Inference Attack via Agentic Self-Exploration</a></h1>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/arXiv-2604.01014-b31b1b.svg)](https://arxiv.org/abs/2604.01014)


</div>

<div align="center">
  Ruhao Liu<sup>&ddagger;</sup>&emsp;Weiqi Huang<sup>&ddagger;</sup>&emsp;Qi Li<sup>&ddagger;</sup>&emsp;Xinchao Wang<sup>&dagger;</sup>
</div>

<div align="center">
    <a href="https://sites.google.com/view/xml-nus/people?authuser=0" target="_blank">xML-Lab</a>, National University of Singapore&emsp;
</div>


<div align="center">
  <sup>&ddagger;</sup>Equal contribution
  <sup>&dagger;</sup>Corresponding author
</div>


## Introduction

Membership Inference Attacks (MIAs) are an important auditing tool for assessing whether a machine learning model leaks information about its training data. However, most existing MIA methods rely on static, handcrafted heuristics, which often lack adaptability and transfer poorly across different large models.

To address this limitation, we propose **AutoMIA**, an agentic framework that reformulates membership inference as an automated process of **self-exploration** and **strategy evolution**. Given high-level scenario specifications, AutoMIA explores the attack space by generating **executable logits-level strategies** and progressively refining them through **closed-loop evaluation feedback**. By decoupling high-level strategy reasoning from low-level execution, AutoMIA enables a systematic and model-agnostic traversal of the attack search space.

Extensive experiments show that AutoMIA consistently matches or outperforms strong baselines while removing the need for manual feature engineering.

## Quick Start

- The detailed implementation is given in: [llava/README.md](llava/README.md)
- Datasets:
  - VL-MIA [image](https://huggingface.co/datasets/JaineLi/VL-MIA-image)
  - VL-MIA [text](https://huggingface.co/datasets/JaineLi/VL-MIA-text)

## Repository Structure

- `agent_mia/` — Agent implementation and configuration
- `llava/` — LLaVA-related scripts and running instructions
  - `run_with_text_agent.py`
  - `run_with_img_agent.py`

## Citation

If you find our work or this codebase helpful, please consider citing our paper:

```bibtex
@article{liu2026automia,
  title={AutoMIA: Improved Baselines for Membership Inference Attack via Agentic Self-Exploration},
  author={Ruhao Liu and Weiqi Huang and Qi Li and Xinchao Wang},
  journal={arXiv preprint arXiv:2604.01014},
  year={2026}
}
```
