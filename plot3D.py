import numpy as np
import plotly.graph_objects as go

def plot_3d_surface_plotly(grid_file, iterates_file, title="3D Surface with Optimization Path"):
    # Load meshgrid data
    grid_data = np.load(grid_file)
    z = grid_data['z']
    x1x2 = grid_data['x1x2']
    x1, x2 = x1x2[..., 0], x1x2[..., 1]  # Separate the meshgrid coordinates

    # Load optimization iterates
    iterates = np.loadtxt(iterates_file, delimiter=",", skiprows=1)
    x0_vals, x1_vals, func_vals = iterates[:, 0], iterates[:, 1], iterates[:, 2]

    # Create Plotly figure
    fig = go.Figure()

    # Add surface plot
    fig.add_trace(go.Surface(
        x=x1, y=x2, z=z,
        colorscale="Viridis",
        opacity=0.4,
        showscale=True
    ))

    # Add scatter plot for optimization path
    fig.add_trace(go.Scatter3d(
        x=x0_vals, y=x1_vals, z=func_vals,
        mode='markers+lines',
        marker=dict(size=5, color='red'),
        line=dict(color='red', width=2),
        name="Optimization Path"
    ))

    # Customize layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            # xaxis=dict(nticks=10, range=[0,11]),
            # yaxis=dict(nticks=10, range=[0,11]),
            #zaxis=dict(nticks=10, range=[0,1.4]),
            zaxis=dict(nticks=10)
        ),
        width=800,
        height=800,
    )

    # Optional: Save as HTML
    fig.write_html("3d_surface_plotly_0.07.html")
    fig.show()

# Usage
plot_3d_surface_plotly("grid_data_lr7.0e-02.npz", "iterate_points_lr0.070.csv")
#plot_3d_surface_plotly("grid_data_holder_lr1.0e-01.npz", "iterate_points_holder_lr0.100.csv")
# import numpy as np
# import plotly.graph_objects as go

# def plot_3d_surface_with_projections(grid_file, iterates_file, title="3D Surface with Optimization Path and Projections"):
#     # Load meshgrid data
#     grid_data = np.load(grid_file)
#     z = grid_data['z']
#     x1x2 = grid_data['x1x2']
#     x1, x2 = x1x2[..., 0], x1x2[..., 1]  # Separate the meshgrid coordinates

#     # Load optimization iterates
#     iterates = np.loadtxt(iterates_file, delimiter=",", skiprows=1)
#     x0_vals, x1_vals, func_vals = iterates[:, 0], iterates[:, 1], iterates[:, 2]

#     # Create Plotly figure
#     fig = go.Figure()

#     # Add surface plot
#     fig.add_trace(go.Surface(
#         x=x1, y=x2, z=z,
#         colorscale="Viridis",
#         opacity=0.3,
#         showscale=True
#     ))

#     # Add 3D optimization path
#     fig.add_trace(go.Scatter3d(
#         x=x0_vals, y=x1_vals, z=func_vals,
#         mode='markers+lines',
#         marker=dict(size=5, color='red'),
#         line=dict(color='red', width=2),
#         name="Optimization Path"
#     ))

#     # Add projection on XY plane (Z=0)
#     fig.add_trace(go.Scatter3d(
#         x=x0_vals, y=x1_vals, z=[0]*len(x0_vals),
#         mode='lines+markers',
#         marker=dict(size=3, color='red'),
#         line=dict(color='red', dash='dash'),
#         name="Projection on XY plane"
#     ))

#     # Add projection on XZ plane (Y=0)
#     fig.add_trace(go.Scatter3d(
#         x=x0_vals, y=[0]*len(x0_vals), z=func_vals,
#         mode='lines+markers',
#         marker=dict(size=3, color='red'),
#         line=dict(color='red', dash='dash'),
#         name="Projection on XZ plane"
#     ))

#     # Add projection on YZ plane (X=0)
#     fig.add_trace(go.Scatter3d(
#         x=[0]*len(x1_vals), y=x1_vals, z=func_vals,
#         mode='lines+markers',
#         marker=dict(size=3, color='red'),
#         line=dict(color='red', dash='dash'),
#         name="Projection on YZ plane"
#     ))

#     # Customize layout
#     fig.update_layout(
#         title=title,
#         scene=dict(
#             xaxis_title='X',
#             yaxis_title='Y',
#             zaxis_title='Z',
#             zaxis=dict(nticks=10, range=[0,1.4]),
#         ),
#         width=800,
#         height=800,
#     )

#     # Save and show
#     fig.write_html("3d_surface_with_projections.html")
#     fig.show()

# # Usage
# plot_3d_surface_with_projections("grid_data.npz", "iterate_points.csv")
