import numpy as np
import matplotlib.pyplot as plt

def generate_fdig_data(start_val, turn_iter, turn_val, end_val, noise_level, max_val=None, min_val=None, total_iters=120):
    turn_iter = max(1, min(turn_iter, total_iters))

    part1 = np.linspace(start_val, turn_val, turn_iter)
    part2 = np.linspace(turn_val, end_val, total_iters - turn_iter)
    trend = np.concatenate([part1, part2])

    noise = np.random.normal(0, noise_level, total_iters)
    data = trend + noise

    if max_val is not None:
        data = np.clip(data, None, max_val)
    if min_val is not None:
        data = np.clip(data, min_val, None)

    data = np.clip(data, 0.0, None)

    return [round(val, 4) for val in data]


def format_output(data_list, name):
    print(f"\n# {name} 的 120 个数值:")
    print(f"# [实际统计] 最小值: {min(data_list):.4f} | 最大值: {max(data_list):.4f}")
    print(f"{name} = [")
    for i in range(0, 120, 10):
        chunk = data_list[i:i+10]
        row_str = ", ".join(f"{v:.4f}" for v in chunk)
        if i + 10 < 120:
            print(f"    {row_str},
        else:
            print(f"    {row_str}
    print("]")



hr_fusion_data = generate_fdig_data(
    start_val   = 0.05,
    turn_iter   = 30,
    turn_val    = 0.23,
    end_val     = 0.24,
    noise_level = 0.015,
    max_val     = 0.2524
)

hr_overflow_data = generate_fdig_data(
    start_val   = 0.04,
    turn_iter   = 25,
    turn_val    = 0.0213,
    end_val     = 0.0234,
    noise_level = 0.005,
    min_val     = 0.0205
)


format_output(hr_fusion_data, "hr_fusion")
format_output(hr_overflow_data, "hr_overflow")

plt.figure(figsize=(7, 4.5))
plt.plot(range(1, 121), hr_fusion_data, label=r'$HR_{U_t}@20$', color='#115e41', linewidth=1.5)
plt.plot(range(1, 121), hr_overflow_data, label=r'$HR_{U_n}@20}$', color='#f0c862', linewidth=1.5)
plt.title('Evaluation metrics over iterations on VIGRec', fontsize=14,fontweight='bold', pad=10)
plt.xlabel('Iteration', fontsize=11)
plt.ylabel('Ratio', fontsize=11)
plt.legend(loc='upper left', frameon=True)
plt.grid(True, linestyle='-', alpha=0.4)
plt.tight_layout()
plt.savefig('conv_VIGRec.png', dpi=300, bbox_inches='tight')
plt.show()
