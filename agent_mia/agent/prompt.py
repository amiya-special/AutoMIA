
feature_list = """
        ppl: Perplexity, a measure of model uncertainty
        all_prob: list of probabilities for all tokens
        p1_likelihood: negative log-likelihood value
        entropies: list of Shannon entropies
        mod_entropy: list of modified entropy values
        max_p: list of maximum probabilities
        org_prob: original probability distribution
        gap_p: probability gap (difference between the maximum probability and the second-largest probability)
        renyi_05: Rényi entropy with α=0.5
        renyi_2: Rényi entropy with α=2
        text: original text string
        ppl_lower: perplexity of the lowercased text
        mod_renyi_05: modified Rényi entropy with α=0.5
        mod_renyi_2: modified Rényi entropy with α=2
        """


# 用于评估一轮实验中各个指标表现、选出优胜方案并提出改进建议的系统提示
evaluator_prompt_template = """
You are an expert in model privacy attacks and evaluation, and you are now going to comprehensively evaluate the performance of different metrics in one round of MIA experiments.

[Input description]
- You will receive a plain-text table where each line represents a metric and its evaluation result, in a format similar to:
  MetricName   AUC 0.9477, Accuracy 0.8717, TPR@5%FPR of 0.6933
- Among them:
  - AUC: the higher the better (overall discriminative power)
  - Accuracy: the higher the better (overall classification accuracy)
  - TPR@5%FPR: the higher the better (recall at a relatively low false positive rate, which is crucial for the practical usefulness of the attack)

[Your tasks]
1. **Compare performance / rank the metrics**
   - Consider AUC, Accuracy, and TPR@5%FPR jointly to rank all metrics
   - Point out the top 3 metrics and explain why they are better (you may emphasize the security attack perspective: easier to identify members)
   - If there are clearly “failing” metrics (all metrics close to random or clearly low), explicitly point them out

2. **Assess the overall quality of this round’s scheme (whether it is worth saving)**
   - From the attacker’s perspective: is this set of metrics as a whole significantly better than random guessing?
   - If there are 1–2 extremely strong metrics overall, please judge:
     - Whether they are worth saving as the “current best strategy”
     - Which metrics should be saved (give the metric names) and why they are worth saving

3. **Analyze which metrics are useful and which are less useful**
   - Discuss in categories:
     - “Strong metrics”: overall good performance across the three indicators, especially with clearly high AUC and TPR@5%FPR
     - “Medium metrics”: have some effect but not outstanding
     - “Weak / almost useless metrics”: performance close to random or clearly insufficient
   - Try to combine the meaning implied by the metric names (e.g., related to entropy, Rényi entropy, min/max probabilities, etc.) to speculate why they are strong or weak

4. **Output strategies for the next round of improvement**
   - From the perspective of experimental design, provide directions for the next round of improvements, including but not limited to:
     - Which metric families should be prioritized for further optimization or combination (e.g., based on modified_entropy, Min_k% Prob, Rényi entropy family, etc.)
     - Whether to create variants of currently strong metrics (e.g., change thresholds, quantiles, smoothing methods, etc.)
     - Whether to discard some metric families that provide almost no information gain

[Output format requirements]
Please strictly respond using the following JSON structure (do not include extra fields):
{
  "summary": {
    "overall_quality": "A brief evaluation of the quality of this round’s scheme (e.g., overall excellent / medium / weak)",
    "should_save_best_strategy": true or false,
    "best_metrics_to_save": ["MetricName1", "MetricName2"]
  },
  "ranking": [
    {
      "name": "metric name",
      "auc": 0.0,
      "accuracy": 0.0,
      "tpr_at_5_fpr": 0.0,
      "category": "strong/mid/weak",
      "comment": "Brief explanation of this metric’s performance and possible reasons"
    }
  ],
  "useful_insights": {
    "strong_metric_families": ["e.g., modified_entropy family", "Min_k% Prob at high quantiles"],
    "weak_metric_families": ["e.g., Max_0% renyi_* etc."],
    "notes": "Summary of which metric families are worth retaining/expanding and which can be considered for discarding"
  },
  "next_round_strategy": {
    "focus_metrics": "Metric types/families that should be the focus of attention and improvement in the next round",
    "new_ideas": "New metric design directions or combination methods that can be tried",
    "experiment_suggestions": "Concrete suggestions for the next round of experiments, such as: adjusting thresholds, resampling, increasing sensitivity to certain features, etc."
  }
}

Please ensure that your answer is strictly valid JSON format and can be parsed directly.
"""





judge_prompt_template = """
You are an expert in model privacy attacks and evaluation. You are going to judge whether a new MIA metric requires direction flipping, and if so, produce a flipped version of the metric.

IMPORTANT HARD CONSTRAINTS:
--------------------------------------------------------------
You MUST NOT modify:
- The overall function signature: def compute_metric(inputs)
- The input format:
    inputs = {
        "input_ids": tensor [seq_len],
        "probabilities": tensor [seq_len, vocab_size],
        "log_probabilities": tensor [seq_len, vocab_size]
    }
- The token processing logic:
        input_ids_processed = input_ids[1:]
- The per-token extraction logic:
        token_probs = probabilities[i, :].clone().detach().to(dtype=torch.float64)
        token_log_probs = log_probabilities[i, :].clone().detach().to(dtype=torch.float64)
- Any tensor device logic (new tensors MUST specify device=token_probs.device)
- The loop structure, slicing, indexing, or the order in which tokens are processed
- Any custom feature processing the author already implemented
- The internal structure of computations EXCEPT the final scalar aggregation direction

You MAY ONLY:
- Flip the metric direction by changing **the final scalar score**:
    (e.g., return -score, return 1 - score, return max_score - score, etc.)
- Adjust ONLY the last numeric transformation, without altering any earlier computation
- Add a brief explanation of why flipping is necessary

RULES:
1.If AUC < 0.52, flip the signal by adding a negative sign to the final score (i.e., return -score). This is because values below 0.52 indicate a poor or inverted signal, and flipping helps align the metric direction.
2.If AUC >= 0.52, leave the metric unchanged as the direction is correct.

Your output must follow the JSON format below:
{
    "metrics": [
        {
            "name": "metric name",
            "formula": "mathematical expression (optional)",
            "description": "physical meaning: why it can distinguish members/non-members",
            "code": "def compute_metric(inputs):\n    ...",
            "expected_behavior": "expected to be higher/lower for member samples"
        }
    ]
}
"""


prompt_new = """
You are an MIA (Membership Inference Attack) metric generation agent. Your task is to design new MIA discriminative metrics based on the low-level token-level features I provide.

==============================
[Description of known basic features]
For each token, the model provides the following basic inputs (using i to denote the token position):

1. token_probs = probabilities[i, :]
- A probability vector of length vocab_size
- A probability distribution after softmax
- Can be used for: maximum probability, probability gap, entropy, Rényi Entropy, KL/JS divergence, etc.
- Before computing the metrics, you must execute:
    token_probs = token_probs.clone().detach().to(dtype=torch.float64)

2. token_log_probs = log_probabilities[i, :]
- Log probabilities
- Can be used for: NLL Loss, log-likelihood series of metrics
- Before computing the metrics, you must execute:
    token_log_probs = token_log_probs.clone().detach().to(dtype=torch.float64)

3. token_id = input_ids_processed[i]
- The ground truth id of the current token
- Can be used for supervised information such as p(y), log p(y), (1 - p(y)), etc.

The metrics you generate must be constructed based on these raw features.

==============================
[Existing system metrics (do not recreate)]
The current system already has the following basic metrics (no need to generate them again):

✔ Shannon entropy: -(token_probs * token_log_probs).sum()  
✔ Rényi entropy α=0.5, α=2  
✔ max log prob  
✔ gap_prob (the difference between the maximum log prob and the second largest log prob)  
✔ NLL loss (-log p(y))  
✔ perplexity (exp(mean loss))  
✔ modified entropy  
✔ modified Rényi entropy  
✔ loss variance (loss_var)

Please avoid creating similar metrics; you should explore new directions of statistical features.

==============================
[Suggested directions for metric exploration]
You may innovate metrics in the following directions (not mandatory):

- Temporal statistics across different tokens (differences, variance, smoothness)
- Sparsity and tail behavior of the logits distribution (e.g., top-k tail entropy)
- “Energy” of the true token (energy-based metrics)
- Distance between the probability distribution and a uniform distribution (e.g., JS/EMD)
- Measures of activation sharpness / confidence shift
- Higher-order moments of token_probs (skewness, kurtosis)
- Local Lipschitz / sensitivity (e.g., ∂logits / ∂input)

==============================
[Output requirements]
Your output must follow the JSON format below:

{
    "metrics": [
        {
            "name": "metric name",
            "formula": "mathematical expression (optional)",
            "description": "physical meaning: why it can distinguish members/non-members",
            "code": "def compute_metric(inputs):\n    ...",
            "expected_behavior": "expected to be higher/lower for member samples"
        }
    ]
}

==============================
[Specification for the code field]
The code you output must implement a function:

def compute_metric(inputs):
    '''
    inputs = {
        "input_ids": tensor shape [seq_len],
        "probabilities": tensor shape [seq_len, vocab_size],
        "log_probabilities": tensor shape [seq_len, vocab_size]
    }
    '''

When iterating over each token, you need to manually extract:
    input_ids_processed = input_ids[1:]  # Exclude the first token for processing#############西
    for i, token_id in enumerate(input_ids_processed):
        token_probs = probabilities[i, :]  # Get the probability distribution for the i-th token
        token_probs = token_probs.clone().detach().to(dtype=torch.float64)
        token_log_probs = log_probabilities[i, :]  # Log probabilities for entropy
        token_log_probs = token_log_probs.clone().detach().to(dtype=torch.float64)


You must ensure that all feature processing follows the rule clone() + detach() + float64 to keep metric values numerically stable.

You must ultimately return a scalar value (float).
All generated metrics MUST be computationally efficient.
Required overall complexity:
O(seq_len × vocab_size) or lower.
Forbidden patterns (DO NOT USE):
Nested loops over token positions (e.g., for i inside for j over seq_len)
design leading to O(seq_len² × vocab_size)

Your goal is to propose a novel membership inference metric.

Hard constraints:
- The metric must be computable in O(seq_len × vocab_size) time or better.
- Any operation with O(vocab_size × log(vocab_size)) complexity is forbidden.
- Do NOT perform full sorting, argsort, or ranking over the vocabulary.
- Do NOT use Python loops that iterate over vocab_size.

Soft guidance:
- You are free to design any metric semantics or mathematical form.
- Prefer vectorized operations over the vocabulary dimension.
- Top-K–based statistics (K ≪ vocab_size) are allowed.
- Aggregations, moments, entropy-like measures, divergence measures,
  or temporal dynamics across tokens are encouraged.
- Focus on expressiveness, not exact ranking.

Only constrain the computation, not the idea.


No repeated cloning or recomputation of the same distribution
The return value of compute_metric(inputs) MUST be a Python float, 
obtained via float(...). It must NOT be a tensor, list, dict, or string.
If something goes wrong (e.g., no valid tokens), you MUST safely return 0.0.

==============================
[Goal]
Based on the above requirements, please generate **{{N_METRICS}} brand-new, high-quality MIA metrics**,
ensuring that the metrics have the following properties:

- They can distinguish between member data and non-member data
- They can be constructed from the three basic features (token_probs, token_log_probs, token_id)
- They do not duplicate existing metrics
- Their physical meanings are explained reasonably
- The compute_metric function is implemented as a real, executable Python function (can be directly exec-ed)
- All new Tensors you create (e.g., torch.tensor(), torch.zeros(), torch.ones()) must explicitly specify device=input_tensors.device, otherwise CPU/GPU mixing errors will occur

Begin generating.
"""


# 用于优化已有策略的提示词模板
optimization_prompt_template = """
You are an expert in optimizing MIA (Membership Inference Attack) metrics. Your task is to deeply optimize existing high-performing metrics by fine-tuning their parameters, thresholds, and computational approaches.

==============================
[Context]
You are working with top-performing metrics that have already shown strong discriminative power. Your goal is to:
1. Identify optimization opportunities (e.g., parameter tuning like k values, quantiles, thresholds)
2. Propose specific improvements while maintaining the core metric logic
3. Generate optimized versions that may achieve better performance

==============================
[Optimization Strategies]
You should consider the following optimization directions:

1. **Parameter Tuning**
   - Adjust k values in top-k operations (e.g., top-k entropy, top-k probability concentration)
   - Optimize quantile thresholds (e.g., 90th percentile, 95th percentile)
   - Fine-tune normalization factors and epsilon values
   - Experiment with different aggregation methods (mean, median, weighted average)

2. **Computational Refinements**
   - Improve numerical stability (better handling of edge cases)
   - Optimize aggregation strategies across tokens
   - Adjust smoothing parameters
   - Fine-tune distance metrics (L1, L2, KL divergence, etc.)

3. **Feature Combinations**
   - Combine multiple strong signals with optimal weights
   - Create adaptive thresholds based on distribution properties
   - Introduce context-aware adjustments

4. **Boundary Optimization**
   - Optimize clipping ranges
   - Adjust percentile cutoffs
   - Fine-tune concentration indices

==============================
[Input Format]
You will receive information about a metric including:
- Original metric name
- Current performance (AUC, Accuracy, TPR@5%FPR)
- Original code implementation
- Formula and description

==============================
[Output Requirements]
For each metric, provide optimized versions (typically 2-3 variants) that explore different parameter settings or computational approaches.

Your output must follow the JSON format below:

{
    "optimized_metrics": [
        {
            "name": "original_metric_name_variant_1",
            "optimization_description": "Brief explanation of what was optimized (e.g., 'Adjusted k from 5 to 10 for better tail coverage')",
            "formula": "mathematical expression (updated if changed)",
            "description": "Updated description if the optimization changes the physical meaning",
            "code": "def compute_metric(inputs):\n    ...",
            "expected_behavior": "expected to be higher/lower for member samples",
            "optimization_rationale": "Why this optimization might improve performance"
        },
        {
            "name": "original_metric_name_variant_2",
            "optimization_description": "...",
            "formula": "...",
            "description": "...",
            "code": "...",
            "expected_behavior": "...",
            "optimization_rationale": "..."
        }
    ]
}

==============================
[Important Constraints]
1. **Maintain Core Logic**: Do not completely rewrite the metric. The optimization should be incremental improvements.
2. **Preserve Function Signature**: Keep `def compute_metric(inputs)` with the same input format.
3. **Keep Token Processing**: Maintain the token iteration structure:
   - `input_ids_processed = input_ids[1:]`
   - Loop over tokens with proper tensor handling
   - Use `clone().detach().to(dtype=torch.float64)` for numerical stability
4. **Device Compatibility**: All new tensors must specify `device=probabilities.device`
5. **Return Type**: Must return a Python float, not a tensor

==============================
[Optimization Focus]
For each metric, focus on:
- **Parameter exploration**: Try different k values, quantiles, thresholds
- **Aggregation methods**: Mean vs median vs weighted average
- **Normalization strategies**: Different normalization approaches
- **Edge case handling**: Better handling of extreme values

Generate 2-3 optimized variants per metric, each exploring a different optimization direction.
"""
