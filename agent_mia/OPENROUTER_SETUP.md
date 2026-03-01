# OpenRouter API 集成说明

## 概述

`BaseAgent` 类现在支持 OpenRouter 平台作为 API 选择。OpenRouter 是一个统一的 API 接口，可以访问多种 AI 模型（如 GPT-4、Claude、Gemini、Llama 等）。

## 快速开始

### 1. 获取 OpenRouter API 密钥

1. 访问 [OpenRouter 官网](https://openrouter.ai)
2. 注册并登录账户
3. 在 API Keys 页面生成新的 API 密钥
4. 复制 API 密钥（格式：`sk-or-v1-...`）

### 2. 配置 config.yaml

编辑 `/root/autodl-tmp/VL-MIA-main/agent_mia/config.yaml` 文件：

```yaml
api:
  type: "openrouter"  # 设置为 openrouter
  api_key: "sk-or-v1-your-api-key-here"  # 替换为你的 OpenRouter API 密钥
  # base_url 会自动设置为 "https://openrouter.ai/api/v1"，也可以手动指定
  http_referer: "https://github.com/VL-MIA"  # 可选：HTTP-Referer 头
  x_title: "VL-MIA Agent"  # 可选：X-Title 头

model:
  name: "openai/gpt-4o"  # OpenRouter 模型格式：provider/model-name
  temperature: 0.6
  max_tokens: 512
```

### 3. 支持的模型格式

OpenRouter 使用 `provider/model-name` 格式指定模型，例如：

- `openai/gpt-4o` - OpenAI GPT-4o
- `openai/gpt-4-turbo` - OpenAI GPT-4 Turbo
- `anthropic/claude-3-opus` - Anthropic Claude 3 Opus
- `anthropic/claude-3-sonnet` - Anthropic Claude 3 Sonnet
- `google/gemini-pro` - Google Gemini Pro
- `meta-llama/llama-3-70b-instruct` - Meta Llama 3 70B
- `deepseek/deepseek-chat` - DeepSeek Chat

查看完整模型列表：https://openrouter.ai/models

## 支持的 API 平台

代码现在支持以下 API 平台：

1. **openrouter** - OpenRouter 平台（支持多种模型）
2. **deepseek** - DeepSeek API
3. **openai** - OpenAI API

## 配置示例

### DeepSeek 配置
```yaml
api:
  type: "deepseek"
  api_key: "sk-your-deepseek-key"
  base_url: "https://api.deepseek.com"

model:
  name: "deepseek-reasoner"
```

### OpenAI 配置
```yaml
api:
  type: "openai"
  api_key: "sk-your-openai-key"
  base_url: "https://api.openai.com/v1"

model:
  name: "gpt-4"
```

### OpenRouter 配置
```yaml
api:
  type: "openrouter"
  api_key: "sk-or-v1-your-key"
  # base_url 会自动设置，也可以手动指定

model:
  name: "openai/gpt-4o"
```

## 代码变更说明

### BaseAgent 类的改进

1. **自动 base_url 设置**：根据 `api.type` 自动设置正确的 API 端点
2. **OpenRouter 特殊处理**：自动添加必需的 HTTP 头（HTTP-Referer 和 X-Title）
3. **向后兼容**：如果未指定 `api.type`，默认使用 DeepSeek

### 使用示例

```python
from agent_mia.agent.base_agent import BaseAgent

# 使用默认配置文件
agent = BaseAgent()

# 或指定自定义配置文件
agent = BaseAgent(config_path="/path/to/your/config.yaml")

# 调用 API
response = agent.ask("Hello, how are you?")
print(response)
```

## 注意事项

1. **API 密钥安全**：请勿将 API 密钥提交到版本控制系统
2. **模型可用性**：OpenRouter 上的某些模型可能需要付费或等待队列
3. **速率限制**：不同模型可能有不同的速率限制
4. **成本**：使用 OpenRouter 时，请关注 API 调用成本

## 故障排除

### 问题：API 调用失败

1. 检查 API 密钥是否正确
2. 确认网络连接正常
3. 查看日志输出中的错误信息
4. 验证模型名称格式是否正确（OpenRouter 需要使用 `provider/model-name` 格式）

### 问题：模型不可用

1. 访问 https://openrouter.ai/models 查看模型状态
2. 某些模型可能需要账户余额或特殊权限
3. 尝试使用其他可用的模型

## 参考资源

- [OpenRouter 官方文档](https://openrouter.ai/docs)
- [OpenRouter 模型列表](https://openrouter.ai/models)
- [OpenRouter API 参考](https://openrouter.ai/docs/api-reference)



