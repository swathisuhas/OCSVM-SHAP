# In visualization/interaction_plotting.py (or similar file)

import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

# Assuming plot_local_explanation_with_interaction might also be in this file

# Define some default colors (similar to shapiq ones)
COLOR_POS = '#ff0d57' # Red for positive contribution (check if this aligns with SHAP library interpretation)
COLOR_NEG = '#1e88e5' # Blue for negative contribution
COLOR_NEUTRAL = '#ffffff' # White/Neutral
COLOR_EDGE = '#aaaaaa' # Default edge color if needed


def plot_local_interaction_network(
    phi_1_group: torch.Tensor,
    interaction_matrix_group: torch.Tensor,
    feature_names: List[str],
    group_id: Optional[str] = None,
    figsize: tuple = (8, 8),
    node_size_scaler: float = 2500.0, # Adjust scaling for node size
    edge_width_scaler: float = 10.0,  # Adjust scaling for edge width
    min_abs_interaction: float = 1e-9 # Threshold to draw an edge
    ):
    """
    Draws an interaction network plot for a single local explanation,
    similar to shapiq's network_plot.

    Nodes represent features, sized/colored by 1st-order SHAP values (phi_1).
    Edges represent pairwise interactions, width/colored by 2nd-order SHAP values (phi_ij).

    Args:
        phi_1_group: 1D Tensor of first-order SHAP values for *all* features
                     for the specific group being explained (shape: [d]).
        interaction_matrix_group: 2D Tensor (d x d) of interaction values
                                  for the specific group being explained.
        feature_names: List of all feature names (length d).
        group_id: An identifier for the group (e.g., its index or a name).
        figsize: Figure size for the plot.
        node_size_scaler: Multiplier to scale node size based on SHAP value magnitude.
        edge_width_scaler: Multiplier to scale edge width based on interaction magnitude.
        min_abs_interaction: Minimum absolute interaction value to draw an edge.
    """
    num_features = phi_1_group.shape[0]

    # --- Input Validation ---
    if interaction_matrix_group.shape != (num_features, num_features):
        raise ValueError("Shape mismatch between phi_1_group and interaction_matrix_group.")
    if len(feature_names) != num_features:
        raise ValueError("Length of feature_names must match number of features.")

    # Convert to numpy for easier processing and NetworkX compatibility
    phi_1_np = phi_1_group.cpu().numpy()
    phi_2_np = interaction_matrix_group.cpu().numpy()

    # --- Create Graph ---
    graph = nx.Graph()
    for i in range(num_features):
        graph.add_node(i, label=feature_names[i])

    # --- Calculate Scaling Range (Based on this group's values) ---
    # Consider magnitudes of both 1st and 2nd order for consistent scaling
    all_phi_1_abs = np.abs(phi_1_np)
    # Extract off-diagonal elements for interaction range calculation
    off_diag_indices = np.triu_indices_from(phi_2_np, k=1)
    all_phi_2_abs = np.abs(phi_2_np[off_diag_indices])

    # Combine non-zero absolute values to find max for scaling
    all_abs_values = np.concatenate([all_phi_1_abs, all_phi_2_abs])
    # Avoid division by zero if all values are zero
    max_abs_value = np.max(all_abs_values) if all_abs_values.size > 0 else 1.0
    if max_abs_value < 1e-9: # Handle case where all values are effectively zero
        max_abs_value = 1.0

    # --- Assign Node Attributes ---
    node_colors = []
    node_sizes = []
    node_edge_colors = []
    for i in range(num_features):
        weight = phi_1_np[i]
        color = COLOR_POS if weight >= 0 else COLOR_NEG
        size = (abs(weight) / max_abs_value) * node_size_scaler
        # Add minimum size to ensure visibility
        node_sizes.append(max(size, node_size_scaler * 0.05))
        node_colors.append(color)
        # Use node color for edge color, slight dimming might be nice but keep simple
        node_edge_colors.append(color)
        # Store the original weight if needed later
        graph.nodes[i]['shap_value'] = weight
        graph.nodes[i]['label'] = feature_names[i]


    # --- Assign Edge Attributes ---
    edge_colors = []
    edge_widths = []
    edges_to_draw = []
    for i in range(num_features):
        for j in range(i + 1, num_features):
            interaction = phi_2_np[i, j]
            if abs(interaction) >= min_abs_interaction:
                edges_to_draw.append((i, j))
                color = COLOR_POS if interaction >= 0 else COLOR_NEG
                width = (abs(interaction) / max_abs_value) * edge_width_scaler
                # Add minimum width
                edge_widths.append(max(width, edge_width_scaler * 0.05))
                edge_colors.append(color)
                # Add edge to graph for layout calculation even if not drawn visibly?
                # Let's add it - networkx handles non-visible edges if width=0
                graph.add_edge(i, j, weight=interaction) # Add edge with weight

    # Filter edge attributes to match edges_to_draw
    # (This ensures lists match if some edges were below threshold)
    # Note: Previous step only added edges if above threshold, so filtering attributes isn't strictly needed here
    # if len(edges_to_draw) != len(edge_colors): # Sanity check
    #     print("Warning: Edge attribute list length mismatch.")


    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    # Arrange nodes in a circle
    pos = nx.circular_layout(graph)

    # Draw Edges
    nx.draw_networkx_edges(
        graph, pos,
        edgelist=edges_to_draw, # Only draw edges above threshold
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.7, # Add some transparency
        ax=ax
    )

    # Draw Nodes
    nodes_drawn = nx.draw_networkx_nodes(
        graph, pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=node_edge_colors, # Add border color based on shap value
        linewidths=1.5, # Slightly thicker border
        ax=ax
    )
    if nodes_drawn: # Set border style
        nodes_drawn.set_edgecolor(node_edge_colors)


    # Draw Labels (position slightly outside the node circle)
    node_labels = nx.get_node_attributes(graph, 'label')
    for node, (x, y) in pos.items():
        # Simple radius calculation (adjust multiplier as needed)
        dist_from_center = np.sqrt(x**2 + y**2)
        if dist_from_center < 1e-6: continue # Skip if node is at origin (unlikely in circular)
        scale_factor = 1.15 # How far out to place labels
        label_x = x * scale_factor
        label_y = y * scale_factor
        # Basic angle calculation for text rotation (optional, can make it complex)
        # angle = np.arctan2(y, x) * (180 / np.pi)
        # if 90 < angle < 270: angle -= 180 # Keep text upright
        # plt.text(label_x, label_y, node_labels[node], ha='center', va='center', rotation=angle, fontsize=9)
        ax.text(label_x, label_y, node_labels[node], ha='center', va='center', fontsize=9)


    title = f"Second-order OCSMM-SHAP Interaction Network for Group {group_id}" if group_id is not None else "Local Interaction Network"
    ax.set_title(title, fontsize=14)
    _add_legend_to_axis(ax)
    plt.tight_layout()
    plt.show()


    # In visualization/interaction_plotting.py

# --- Add these constants at the top of the file ---
COLOR_POS = '#ff0d57' # Red for positive contribution (Update if your interpretation differs!)
COLOR_NEG = '#1e88e5' # Blue for negative contribution
# ---

# --- Your plot_local_interaction_network function definition here ---
# ...

# --- Add this helper function (adapted from shapiq) ---
def _add_legend_to_axis(axis: plt.Axes) -> None:
    """
    Adds legends for order 1 (nodes) and order 2 (edges) interactions
    to the specified axis, similar to shapiq style.

    Args:
        axis: The matplotlib Axes object to add the legend to.
    """
    # Define fixed sizes/labels for legend entries
    sizes = [1.0, 0.2, 0.2, 1]  # Represents relative magnitude for legend examples
    labels = ["High Positive", "Low Positive", "Low Negative", "High Negative"] # Clearer labels
    alphas_line = [0.8, 0.4, 0.4, 0.8] # Adjusted alpha for better visibility

    # Create dummy plot objects for legend handles (Order 1 - Nodes)
    plot_circles = []
    for i in range(4):
        size = sizes[i]
        color = COLOR_POS if i < 2 else COLOR_NEG # Match plot colors
        # Plot empty data just to get a handle with the right style
        circle = axis.plot([], [], c=color, marker="o", markersize=size * 8, linestyle="None")
        plot_circles.append(circle[0])

    # Get default legend fontsize or set a default
    try:
        font_size = plt.rcParams["legend.fontsize"]
    except KeyError:
        font_size = 'small' # Default if not set in rcParams

    # Create the first legend for node contributions
    legend1 = axis.legend(
        plot_circles,
        labels,
        frameon=True,
        framealpha=0.7, # Slightly more opaque background
        facecolor="white",
        title=r"$\bf{Feature Effect}$" + "\n(Node Size/Color)", # Updated Title
        fontsize=font_size,
        labelspacing=0.5,
        handletextpad=0.5,
        borderpad=0.5,
        handlelength=1.5,
        title_fontsize=font_size,
        loc="upper left", # Original position
        bbox_to_anchor=(-1.1, 1.0) # Adjust anchor slightly left if needed
    )

    # Create dummy plot objects for legend handles (Order 2 - Edges)
    plot_lines = []
    for i in range(4):
        size = sizes[i]
        alpha = alphas_line[i]
        color = COLOR_POS if i < 2 else COLOR_NEG # Match plot colors
        # Plot empty data just to get a handle with the right style
        line = axis.plot([], [], c=color, linewidth=size * 3, alpha=alpha)
        plot_lines.append(line[0])

    # Create the second legend for edge interactions
    legend2 = axis.legend(
        plot_lines,
        labels,
        frameon=True,
        framealpha=0.7,
        facecolor="white",
        title=r"$\bf{Interaction Effect}$" + "\n(Edge Width/Color)", # Updated Title
        fontsize=font_size,
        labelspacing=0.5,
        handletextpad=0.5,
        borderpad=0.5,
        handlelength=1.5,
        title_fontsize=font_size,
        loc="upper right", # Original position
        bbox_to_anchor=(1.1, 1.0) # Adjust anchor slightly right if needed
    )

    # Add both legends to the axis so they don't overwrite each other
    axis.add_artist(legend1)
    # axis.add_artist(legend2) # Add this back if the second legend is desired and fits