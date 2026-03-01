# metrics_plugin.py

import types

METRICS_REGISTRY = {}

def register_metric(name, code_str):
    """
    动态加载来自 JSON 的代码片段，将其变成可调用函数。
    code_str 必须定义一个名为 compute_metric(inputs) 的函数。
    """
    module = types.ModuleType(name)
    exec(code_str, module.__dict__)  # 动态载入代码

    if "compute_metric" not in module.__dict__:
        raise ValueError(f"Metric {name} must define compute_metric(inputs)")

    METRICS_REGISTRY[name] = module.compute_metric
    print(f"[METRIC REGISTERED] {name}")

def compute_custom_metrics(inputs):
    results = {}
    for name, func in METRICS_REGISTRY.items():
        try:
            results[name] = func(inputs)
        except Exception as e:
            results[name] = f"ERROR: {e}"
    return results
