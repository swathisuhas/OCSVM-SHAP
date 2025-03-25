import shap
import torch

def local_bar_plot(shapley_values: torch.Tensor, index:int, **kwargs):
    title = f"SHAP value (Local Explanation) for index {index}"
    return shap.bar_plot(shapley_values.T[index], show=True, max_display=10, string = title,**kwargs)

def global_bar_plot(shapley_values: torch.Tensor, **kwargs):
    return shap.bar_plot(shapley_values.T, show=True, max_display=10, string = "Absolute Mean SHAP value (Global Explanation)",
                             **kwargs)