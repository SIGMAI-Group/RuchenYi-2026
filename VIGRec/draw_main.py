import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / 'draw'))
from svg_export import enable_editable_svg_text
from matplotlib.ticker import FormatStrFormatter

enable_editable_svg_text()

import matplotlib.pyplot as plt

metric_1_name = r'$HR_{U_t}@20$'
metric_2_name = r'$HR_{U_n}@20$'

methods = ['Clean', 'Random', 'VIGRec(WMF-SGD)', 'VIGRec(ItemAE)']

DATA = {
    'LightGCN': {
        'HR_Ut': [0.0002, 0.0033, 0.0553, 0.0599],
        'HR_Un': [0.0001, 0.0027, 0.0120, 0.0145]
    },
    'LATTICE': {
        'HR_Ut': [0.0001, 0.0145, 0.1535, 0.1497],
        'HR_Un': [0.0001, 0.0118, 0.0235, 0.0175]
    },
    'DualGNN': {
        'HR_Ut': [0.0001, 0.0230, 0.2209, 0.2630],
        'HR_Un': [0.0002, 0.0225, 0.0175, 0.0165]
    },
    'SLMRec': {
        'HR_Ut': [0.0000, 0.0506, 0.1359, 0.1953],
        'HR_Un': [0.0000, 0.0479, 0.0322, 0.0524]
    },
    'LGMRec': {
        'HR_Ut': [0.0001, 0.0762, 0.2524, 0.2486],
        'HR_Un': [0.0002, 0.0807, 0.0205, 0.0122]
    },
    'HMCRec': {
        'HR_Ut': [0.0001, 0.0426, 0.1855, 0.1889],
        'HR_Un': [0.0001, 0.0410, 0.0135, 0.0253]
    }
}

target_models = list(DATA.keys())
n_models = len(target_models)
n_methods = len(methods)

n_cols = 3
n_rows = int(np.ceil(n_models / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 5.4 * n_rows))
axes = np.array(axes).reshape(-1)

bar_width = 0.35
index = np.arange(n_methods)

color_ut = '#48D1CC' # 亮蓝绿色 (Turquoise)
color_un = '#1E90FF' # 湖蓝色 (DodgerBlue)

for i, model in enumerate(target_models):
    ax = axes[i]

    hr_ut = DATA[model]['HR_Ut']
    hr_un = DATA[model]['HR_Un']

    rects1 = ax.bar(index - bar_width/2, hr_ut, bar_width, label=metric_1_name, color=color_ut, edgecolor='grey', linewidth=0.5)
    rects2 = ax.bar(index + bar_width/2, hr_un, bar_width, label=metric_2_name, color=color_un, edgecolor='grey', linewidth=0.5)

    ax.set_title(model, fontweight='bold', fontsize=15)
    ax.set_xticks(index)
    ax.set_xticklabels(methods, fontsize=8, rotation=32, ha='right', rotation_mode='anchor')

    local_max = max(max(hr_ut), max(hr_un))

    if local_max < 0.001:
        local_max = 0.001
    ax.set_ylim(0, local_max * 1.45)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    if i % n_cols == 0:
        ax.set_ylabel('Ratio (HR@20)', fontsize=13)

    ax.legend(handles=[rects1, rects2], labels=[metric_1_name, metric_2_name], loc='upper left', fontsize=10, framealpha=0.95, edgecolor='grey')

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

for j in range(n_models, len(axes)):
    axes[j].axis('off')

plt.tight_layout(pad=1.2, h_pad=3.0, w_pad=1.4)

plt.savefig('fusion_performance_comparison.pdf', format='pdf', bbox_inches='tight')
plt.savefig('fusion_performance_comparison.svg', format='svg', dpi=300, bbox_inches='tight')

print("绘图完成！已保存为 fusion_performance_comparison.pdf 和 .svg")
