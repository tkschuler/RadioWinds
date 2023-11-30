import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

def create_continuous_flow_field(observations, grid_size=(50, 50)):
    # Determine the size of the grid
    rows, cols = grid_size

    # Create an empty grid for the flow field
    flow_field = np.zeros((rows, cols, 2))

    # Calculate the grid cell size
    cell_width = 1.0 / (rows - 1)
    cell_height = 1.0 / (cols - 1)

    # Extract observations into separate arrays for interpolation
    x_obs, y_obs, dx_obs, dy_obs = np.split(observations, 4, axis=1)

    print(x_obs)

    # Create a bivariate spline for interpolation
    spline_dx = RectBivariateSpline(x_obs.flatten(), y_obs.flatten(), dx_obs.flatten(), kx=1, ky=1)
    spline_dy = RectBivariateSpline(x_obs.flatten(), y_obs.flatten(), dy_obs.flatten(), kx=1, ky=1)

    # Interpolate the flow field values for each grid cell
    for i in range(rows):
        for j in range(cols):
            x = i * cell_width
            y = j * cell_height
            flow_field[i, j, 0] = spline_dx(x, y)
            flow_field[i, j, 1] = spline_dy(x, y)

    # Normalize the flow vectors
    magnitudes = np.linalg.norm(flow_field, axis=2, keepdims=True)
    flow_field /= np.maximum(magnitudes, 1e-8)

    # Create a 2D grid for plotting
    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)
    X, Y = np.meshgrid(x, y)

    # Plot the flow field using quiver plot
    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, flow_field[:, :, 0], flow_field[:, :, 1], pivot='mid', scale=25, color='b')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Continuous Flow Field')
    plt.show()

# Example usage:
np.random.seed(0)  # Setting a seed for reproducibility
observations = np.random.rand(100, 4)  # Generate random continuous space observations (x, y, dx, dy)
observations[:, :2] = np.sort(observations[:, :2], axis=0)  # Sort x and y values

create_continuous_flow_field(observations)
