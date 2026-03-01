base_prompt_template = """
        你是一个专门为成员推理攻击(MIA)设计评估指标的专家。你的任务是基于模型输出的统计特征，
        创建新的、有效的指标来区分训练集成员和非成员样本。
        
        已知信息：
        1. 成员样本通常具有更低的熵、更高的置信度和更集中的概率分布
        2. 非成员样本通常具有更高的熵、更低的置信度和更分散的概率分布
        3. 我们有以下基础特征可用：
        {feature_list}
        4. 这是我们已知的模型输入特征：
        token_probs, token_log_probs，token_id,你的函数可以选择性基于这些输入特征来计算指标。
        example:
        def modified_entropy(token_prob, token_id):
            token_probs_safe = torch.clamp(token_probs, min=epsilon, max=1-epsilon)
            p_y = token_probs_safe[token_id].item()
            modified_entropy = -(1 - p_y) * torch.log(torch.tensor(p_y)) - (token_probs * torch.log(1 - token_probs_safe)).sum().item() + p_y * torch.log(torch.tensor(1 - p_y)).item()
            modified_entropies.append(modified_entropy)
        
        要求：
        1. 创建新的、创新的指标计算公式，这些公式应该是基础特征的组合或变换
        2. 每个指标都应该有明确的数学表达式和物理意义解释
        3. 指标名称应该简洁明了，反映其计算逻辑
        4. 提供Python代码来计算该指标
        5. 尝试创建多样化的指标类型（比率、差值、乘积、非线性组合等）

        
        请以以下JSON格式输出：
        {{
            "metrics": [
                {{
                    "name": "指标名称",
                    "formula": "数学表达式",
                    "description": "物理意义解释",
                    "code": "计算该指标的Python代码片段",
                    "expected_behavior": "成员样本的指标值预期会高于/低于非成员样本"
                }}
            ]
        }}
        """
feature_list = """
        ppl: 困惑度 (Perplexity)，模型不确定性的度量
        all_prob: 所有token的概率列表
        p1_likelihood: 负对数似然值
        entropies: Shannon熵列表
        mod_entropy: 修改后的熵值列表
        max_p: 最大概率列表
        org_prob: 原始概率分布
        gap_p: 概率差距（最大概率与次大概率之差）
        renyi_05: α=0.5的Renyi熵
        renyi_2: α=2的Renyi熵
        text: 原始文本字符串
        ppl_lower: 小写文本的困惑度
        mod_renyi_05: 修改后的α=0.5的Renyi熵
        mod_renyi_2: 修改后的α=2的Renyi熵
        """

# 用于评估一轮实验中各个指标表现、选出优胜方案并提出改进建议的系统提示
evaluator_prompt_template = """
你是一名精通模型隐私攻击与评估的专家，现在要对一轮 MIA 实验中不同指标的表现进行综合评估。

【输入说明】
- 你会收到一段纯文本表格，每一行代表一个指标及其评价结果，格式类似：
  指标名   AUC 0.9477, Accuracy 0.8717, TPR@5%FPR of 0.6933
- 其中：
  - AUC 越高越好（整体区分能力）
  - Accuracy 越高越好（总体判别精度）
  - TPR@5%FPR 越高越好（在较低误报率下的召回能力，对攻击实用性很关键）

【你的任务】
1. **比较胜负 / 排序指标**
   - 综合考虑 AUC、Accuracy、TPR@5%FPR，对所有指标进行排序
   - 指出表现最好的 3 个指标，并解释为什么它们更好（可侧重于安全攻击视角：更容易识别成员）
   - 如果有明显“失败”的指标（各项指标接近随机或明显偏低），请点名指出

2. **判断本轮方案总体质量（是否值得保存）**
   - 从“攻击者角度”评价：这组指标整体上是否显著优于随机猜测？
   - 如果整体上存在 1~2 个非常突出的指标，请判断：
     - 是否值得作为“当前最优策略”保存？
     - 建议保存哪些指标（给出指标名称）以及为什么值得保存

3. **分析哪些指标有用、哪些不太有用**
   - 分类讨论：
     - “强指标”：在三项指标中整体表现较好，尤其是 AUC 和 TPR@5%FPR 明显高
     - “中等指标”：有一定效果，但优势不突出
     - “弱指标 / 基本无用”：表现接近随机或明显不足
   - 尽量结合指标名称所隐含的含义（如与熵、Renyi 熵、min/max 概率等相关），推测为什么它们会强或弱

4. **输出下一轮改进策略**
   - 从实验设计的角度，给出下一轮可以尝试的改进方向，包括但不限于：
     - 应该重点继续优化或组合的指标类别（例如基于 modified_entropy、Min_k% Prob、Renyi 熵类等）
     - 是否可以对当前强指标再进行变体（例如改变阈值、分位点、平滑方式等）
     - 是否要丢弃某些几乎没有信息增益的指标家族

【输出格式要求】
请严格使用以下 JSON 结构回复（不要包含多余的字段）：
{
  "summary": {
    "overall_quality": "简要评价本轮方案质量（例如：整体优秀/中等/较弱）",
    "should_save_best_strategy": true 或 false,
    "best_metrics_to_save": ["指标名1", "指标名2"]
  },
  "ranking": [
    {
      "name": "指标名称",
      "auc": 0.0,
      "accuracy": 0.0,
      "tpr_at_5_fpr": 0.0,
      "category": "strong/mid/weak",
      "comment": "简要说明该指标表现和可能原因"
    }
  ],
  "useful_insights": {
    "strong_metric_families": ["例如：modified_entropy 系列", "Min_k% Prob 在高分位数"],
    "weak_metric_families": ["例如：Max_0% renyi_* 等"],
    "notes": "关于哪些指标家族值得保留/扩展、哪些可以考虑舍弃的总结"
  },
  "next_round_strategy": {
    "focus_metrics": "下一轮应该重点关注和改进的指标类型/家族",
    "new_ideas": "可以尝试的新的指标设计方向或组合方式",
    "experiment_suggestions": "下一轮实验设置的具体建议，例如：调整阈值、重新采样、增加对某些特征的敏感性等"
  }
}

请确保你的回答严格是合法 JSON 格式，可以直接被解析。
"""
prompt_new = """
你是一个 MIA（Membership Inference Attack）指标生成智能体，你的任务是基于我提供的
底层 token-level 特征，设计新的 MIA 判别指标。

==============================
【已知基础特征说明】
模型为每个 token 提供如下基础输入（以 i 表示 token 位置）：

1. token_probs = probabilities[i, :]
- 一个长度为 vocab_size 的概率向量
- 经过 softmax 后的概率分布
- 可用于：最大概率、概率差距、熵、Rényi Entropy、KL/JS divergence 等
- 在计算指标前应执行：
    token_probs = token_probs.clone().detach().to(dtype=torch.float64)

2. token_log_probs = log_probabilities[i, :]
- 对数概率
- 可用于：NLL Loss, log-likelihood 系列指标
- 在计算指标前应执行：
    token_log_probs = token_log_probs.clone().detach().to(dtype=torch.float64)

3. token_id = input_ids_processed[i]
- 当前 token 的 ground truth id
- 可用于：p(y), log p(y), (1 - p(y)) 等有监督信息

你生成的指标必须基于这些原始特征构造。

==============================
【系统已有指标（不要重复创造）】
当前系统已经拥有以下基础指标（无需再生成）：

✔ Shannon entropy：-(token_probs * token_log_probs).sum()  
✔ Rényi entropy α=0.5, α=2  
✔ max log prob  
✔ gap_prob（最大 log prob 与第二大 log prob 差）  
✔ NLL loss（-log p(y)）  
✔ perplexity（exp(mean loss)）  
✔ modified entropy  
✔ modified Rényi entropy  
✔ loss variance（loss_var）

请避免重复类似指标，应该往新的统计特征方向探索。

==============================
【指标探索方向建议】
你可以在以下方向创新指标（不是必须）：

- 不同 token 间的时间序列统计（差分、方差、平滑度）
- logits 分布的稀疏性、尾部分布（例如 top-k tail entropy）
- 对真实 token 的“能量值”（energy-based metrics）
- 概率分布与均匀分布的距离（如 JS/EMD）
- 激活 sharpness / confidence-shift 度量
- 使用 token_probs 的高阶矩（skewness, kurtosis）
- 局部 Lipschitz / sensitivity（如 ∂logits / ∂input）

==============================
【输出要求】
你的输出必须遵循以下 JSON 格式：

{
    "metrics": [
        {
            "name": "指标名称",
            "formula": "数学表达式（可选）",
            "description": "物理意义：为什么它能区别成员/非成员",
            "code": "def compute_metric(inputs):\n    ...",
            "expected_behavior": "成员样本预计更高/更低"
        }
    ]
}

==============================
【code 字段规范】
你输出的 code 必须实现一个函数：

def compute_metric(inputs):
    '''
    inputs = {
        "input_ids": tensor shape [seq_len],
        "probabilities": tensor shape [seq_len, vocab_size],
        "log_probabilities": tensor shape [seq_len, vocab_size]
    }
    '''

在遍历每个 token 时，你需要自行提取：
    input_ids_processed = input_ids[1:]  # Exclude the first token for processing#############西
    for i, token_id in enumerate(input_ids_processed):
        token_probs = probabilities[i, :]  # Get the probability distribution for the i-th token
        token_probs = token_probs.clone().detach().to(dtype=torch.float64)
        token_log_probs = log_probabilities[i, :]  # Log probabilities for entropy
        token_log_probs = token_log_probs.clone().detach().to(dtype=torch.float64)


你必须保证所有特征处理遵循 clone() + detach() + float64 的规则，使指标数值稳定。

最终必须返回标量类型（float）。

==============================
【目标】
请基于以上要求，生成 **1~3 个全新的高质量 MIA 指标**，
确保指标具有以下特点：

- 对成员数据与非成员数据具有可区分性
- 能够基于三大基础特征构造（token_probs, token_log_probs, token_id）
- 不与现有指标重复
- 解释物理意义合理
- 用 compute_metric 写出真实可执行的 python 函数（可直接 exec 运行）
- 所有你创建的新 Tensor（例如 torch.tensor()、torch.zeros()、torch.ones() ）必须显式指定 device=input_tensors.device，否则会导致 CPU/GPU 混合报错

开始生成。
"""
