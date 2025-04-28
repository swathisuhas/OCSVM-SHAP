import shap
import torch
import numpy as np
import matplotlib.pyplot as plt

def local_bar_plot_ocsvm(shapley_values: torch.Tensor, index:int, feature_names, dataset, **kwargs):
    explanation = shap.Explanation(
        values=shapley_values[index],
        data=dataset[index],
        feature_names=feature_names
    )
    # Plot
    title = f"SHAP value (Local Explanation) for index {index}"
    shap.plots.bar(explanation, show=False, max_display=5)
    plt.xlabel(title)
    plt.show()

def global_bar_plot_ocsvm(shapley_values: torch.Tensor,feature_names, dataset, **kwargs):
    explanation = shap.Explanation(
        values=shapley_values,
        data=np.mean(dataset, axis=0),
        feature_names=feature_names
    )
    # Plot
    title = f"Absolute Mean SHAP value (Global Explanation)"
    shap.plots.bar(explanation, show=False, max_display=20)
    plt.xlabel(title)
    plt.show()

def summary_plot_ocsvm(shapley_values: torch.Tensor, **kwargs):
    shap.summary_plot(shapley_values, show=False, max_display=5, **kwargs)
    plt.title("Beeswarm summary plot")
    plt.xlabel("SHAP value (impact on model output)")
    plt.show

def summary_plot_ocsmm(shapley_values: torch.Tensor, feature_names, datasets, **kwargs):
    # Datasets is a list of groups: each group is a list of samples
    group_data = [np.mean(group, axis=0) for group in datasets]
    group_data = np.array(group_data)

    explanation = shap.Explanation(
        values=shapley_values,
        feature_names=feature_names,
        data = group_data
    )

    # Now using shap.plots.summary instead of shap.summary_plot
    shap.plots.violin(explanation, show=False, max_display=5, **kwargs)

    plt.title("Violin summary plot")
    plt.xlabel("SHAP value (impact on model output)")
    plt.show()


def local_bar_plot_ocsmm(shapley_values: torch.Tensor, index:int, feature_names, datasets, **kwargs):
    # Datasets is a list of groups: each group is a list of samples
    group_data = [np.mean(group, axis=0) for group in datasets]
    group_data = np.array(group_data)

    # Build Explanation object for the selected index
    explanation = shap.Explanation(
        values=shapley_values[index],
        data=group_data[index],
        feature_names=feature_names
    )

    # Plot
    title = f"SHAP value (Local Explanation) for index {index}"
    shap.plots.bar(explanation, show=False, max_display=5)
    plt.xlabel(title)
    plt.show()


def global_bar_plot_ocsmm(shapley_values: torch.Tensor, feature_names, datasets, **kwargs):
    # Mean across all groups
    mean_data = np.mean([np.mean(group, axis=0) for group in datasets], axis=0)

    explanation = shap.Explanation(
        values=shapley_values,        
        data=mean_data,                
        feature_names=feature_names
    )

    shap.plots.bar(explanation, show=False, max_display=5, **kwargs)
    plt.xlabel("Absolute Mean SHAP value (Global Explanation)")
    # plt.tight_layout()
    plt.show()