import os
import argparse
from utils.quick_start import quick_start
import numpy as np
os.environ['NUMEXPR_MAX_THREADS'] = '48'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MMRec Attack Evaluation Launcher")
    parser.add_argument('--model', '-m', type=str, default='vbpr', help='测试的受害者多模态模型名称')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='数据集名称')
    parser.add_argument('--fake_data', '-f', type=str, required=True, help='Attack框架生成的 best.npz 假数据路径')
    parser.add_argument('--target_file', '-t', type=str,default='sampled_target_items_5_head.npz',required=True, help='Attack框架生成的 sampled_target_items_xxx.npz 文件路径')
    args = parser.parse_args()

    loaded_targets = np.load(args.target_file)['target_items'].tolist()
    config_dict = {
        'gpu_id': 0,

        'fake_data_path': args.fake_data,

        'target_items': loaded_targets,

        'topk': [20, 50],

        'eval_batch_size': 2048
    }

    print("="*50)
    print(f"🚀 开始多模态对抗攻击测试...")
    print(f"🔪 攻击目标物品: {config_dict['target_items']}")
    print(f"📦 毒药数据路径: {config_dict['fake_data_path']}")
    print(f"🛡️ 正在遭受攻击的模型: {args.model}")
    print("="*50)

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=False)
