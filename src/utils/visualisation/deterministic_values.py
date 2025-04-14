import shap
import torch
import matplotlib.pyplot as plt

def local_bar_plot(shapley_values: torch.Tensor, index:int, **kwargs):
    title = f"SHAP value (Local Explanation) for index {index}"
    shap.bar_plot(shapley_values.T[index], show=False, max_display=20, **kwargs)
    plt.xlabel(title)
    plt.show

def global_bar_plot(shapley_values: torch.Tensor, **kwargs):
    shap.bar_plot(shapley_values.T, show=False, max_display=20,**kwargs)
    plt.xlabel("Absolute Mean SHAP value (Global Explanation)")
    plt.show()

def summary_plot(shapley_values: torch.Tensor, **kwargs):
    shap.summary_plot(shapley_values, show=False, **kwargs)
    plt.title("Beeswarm summary plot")
    plt.xlabel("SHAP value (impact on model output)")
    plt.show