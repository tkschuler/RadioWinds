import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Create a ScalarMappable with a colormap and normalization range
cmap = plt.get_cmap('seismic')  # Red-to-blue reversed
norm = mpl.colors.Normalize(vmin=-100, vmax=100)

# Dummy mappable for colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # No image, just colorbar

# Create a figure only for the colorbar
fig, ax = plt.subplots(figsize=(6, 1))  # Wide and short
fig.subplots_adjust(bottom=0.5)

# Create horizontal colorbar
cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
cbar.set_label("Difference in Probability (in %)")

plt.show()

# Save to file
fig.savefig('colorbar.png', dpi=300, bbox_inches='tight', transparent=True)
plt.close(fig)

print("Saved horizontal colorbar to 'colorbar.png'")
