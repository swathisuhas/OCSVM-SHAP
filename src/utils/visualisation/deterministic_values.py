import shap
import torch


def summary_plot(shapley_values: torch.Tensor,
                 query_data: torch.Tensor,
                 **kwargs
                 ):
    return shap.summary_plot(shapley_values.T.numpy(), query_data.numpy(), show=False,
                             **kwargs)


def violin_plot(shapley_values: torch.Tensor,
                 query_data: torch.Tensor,
                 **kwargs
                 ):
    return shap.violin_plot(shapley_values.T.numpy(), query_data.numpy(), show=False,
                             **kwargs)


def bar_plot(shapley_values: torch.Tensor,
                 query_data: torch.Tensor,
                 **kwargs
                 ):
    return shap.bar_plot(shapley_values.T.numpy(), query_data.numpy(), show=False,
                             **kwargs)