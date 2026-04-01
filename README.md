# AutoMIA

Codebase for **AutoMIA**, an agentic framework for **Vision-Language Membership Inference Attacks (VL-MIA)**.

## Introduction

Membership Inference Attacks (MIAs) are an important auditing tool for assessing whether a machine learning model leaks information about its training data. However, most existing MIA methods rely on static, handcrafted heuristics, which often lack adaptability and transfer poorly across different large models.

To address this limitation, we propose **AutoMIA**, an agentic framework that reformulates membership inference as an automated process of **self-exploration** and **strategy evolution**. Given high-level scenario specifications, AutoMIA explores the attack space by generating **executable logits-level strategies** and progressively refining them through **closed-loop evaluation feedback**. By decoupling high-level strategy reasoning from low-level execution, AutoMIA enables a systematic and model-agnostic traversal of the attack search space.

Extensive experiments show that AutoMIA consistently matches or outperforms strong baselines while removing the need for manual feature engineering.

## Quick Start

- For detailed instructions, please refer to: [llava/README.md](llava/README.md)
- Datasets:
  - VL-MIA [image](https://huggingface.co/datasets/JaineLi/VL-MIA-image)
  - VL-MIA [text](https://huggingface.co/datasets/JaineLi/VL-MIA-text)

## Repository Structure

- `agent_mia/` — Agent implementation and configuration
- `llava/` — LLaVA-related scripts and running instructions
  - `run_with_text_agent.py`
  - `run_with_img_agent.py`

## License

Please choose and add an appropriate `LICENSE` file for your release.
