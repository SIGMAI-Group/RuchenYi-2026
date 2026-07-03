import numpy as np
from svg_export import enable_editable_svg_text

enable_editable_svg_text()

import matplotlib.pyplot as plt

epochs = np.arange(0, 121, 20)

baby_target = np.array([0.063, 0.165, 0.224, 0.245, 0.252, 0.255, 0.254])
baby_nontarget = np.array([0.071, 0.038, 0.022, 0.019, 0.020, 0.018, 0.021])

clothing_target = np.array([0.053, 0.085, 0.132, 0.155, 0.168, 0.175, 0.178])
clothing_nontarget = np.array([0.041, 0.032, 0.033, 0.035, 0.036, 0.039, 0.038])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5))

color_target = '#1f77b4'
color_nontarget = '#ff7f0e'
marker_size = 8
line_width = 2

axes[0].plot(epochs, baby_target, marker='o', markersize=marker_size, linewidth=line_width,
             color=color_target, label=r'$HR_{Ut}@20$ (Targeted Hit ratio)')
axes[0].plot(epochs, baby_nontarget, marker='s', markersize=marker_size, linewidth=line_width,
             color=color_nontarget, label=r'$HR_{Un}@20$ (Non-target Hit ratio)')

axes[0].set_title('Evaluation metrics over epochs on Amazon Baby', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Ratio (HR@20)', fontsize=11)
axes[0].set_xticks(epochs)
axes[0].set_ylim([-0.02, 0.35])

axes[0].legend(loc='upper right')
axes[0].grid(True, linestyle='--', alpha=0.5)


axes[1].plot(epochs, clothing_target, marker='o', markersize=marker_size, linewidth=line_width,
             color=color_target, label=r'$HR_{Ut}@20$ (Targeted Hit ratio)')
axes[1].plot(epochs, clothing_nontarget, marker='s', markersize=marker_size, linewidth=line_width,
             color=color_nontarget, label=r'$HR_{Un}@20$ (Non-target Hit ratio)')

axes[1].set_title('Evaluation metrics over epochs on Amazon Clothing', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Ratio (HR@20)', fontsize=11)
axes[1].set_xticks(epochs)
axes[1].set_ylim([-0.02, 0.35])

axes[1].legend(loc='upper right')
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('epoch_20_comparison_final.svg', dpi=300, bbox_inches='tight')
