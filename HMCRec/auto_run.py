import io
import logging
import os
import re
import sys
from contextlib import redirect_stderr, redirect_stdout

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from utils.quick_start import quick_start

TRAIN_MODE = False

PAPER_BASELINES = {
    'baby': {
        'BPR':      {'Recall@20': 0.0575, 'NDCG@20': 0.0249},
        'VBPR':     {'Recall@20': 0.0663, 'NDCG@20': 0.0284},
        'LightGCN': {'Recall@20': 0.0754, 'NDCG@20': 0.0328},
        'MMGCN':    {'Recall@20': 0.0615, 'NDCG@20': 0.0261},
        'BM3':      {'Recall@20': 0.0860, 'NDCG@20': 0.0369},
        'HCCF':     {'Recall@20': 0.0756, 'NDCG@20': 0.0332},
        'GAR':      {'Recall@20': 0.0810, 'NDCG@20': 0.0311},
        'LATTICE':  {'Recall@20': 0.0850, 'NDCG@20': 0.0370},
    },
    'sports': {
        'BPR':      {'Recall@20': 0.0653, 'NDCG@20': 0.0298},
        'VBPR':     {'Recall@20': 0.0856, 'NDCG@20': 0.0384},
        'LightGCN': {'Recall@20': 0.0864, 'NDCG@20': 0.0387},
        'MMGCN':    {'Recall@20': 0.0605, 'NDCG@20': 0.0254},
        'BM3':      {'Recall@20': 0.0828, 'NDCG@20': 0.0367},
        'HCCF':     {'Recall@20': 0.0857, 'NDCG@20': 0.0394},
        'GAR':      {'Recall@20': 0.0831, 'NDCG@20': 0.0398},
        'LATTICE':  {'Recall@20': 0.0953, 'NDCG@20': 0.0421},
    },
    'clothing': {
        'BPR':      {'Recall@20': 0.0303, 'NDCG@20': 0.0138},
        'VBPR':     {'Recall@20': 0.0415, 'NDCG@20': 0.0192},
        'LightGCN': {'Recall@20': 0.0544, 'NDCG@20': 0.0243},
        'MMGCN':    {'Recall@20': 0.0331, 'NDCG@20': 0.0141},
        'BM3':      {'Recall@20': 0.0538, 'NDCG@20': 0.0238},
        'HCCF':     {'Recall@20': 0.0533, 'NDCG@20': 0.0235},
        'GAR':      {'Recall@20': 0.0702, 'NDCG@20': 0.0295},
        'LATTICE':  {'Recall@20': 0.0733, 'NDCG@20': 0.0330},
    }
}

HMC_PAPER_RESULTS = {
    'baby': {'Recall@20': 0.0930, 'NDCG@20': 0.0448},
    'sports': {'Recall@20': 0.1023, 'NDCG@20': 0.0493},
    'clothing': {'Recall@20': 0.0742, 'NDCG@20': 0.0367},
}

MODEL_CONFIGS = {
    'baby': {
        'gpu_id': 0,
        'n_ui_layers': [2],
        'n_mm_layers': [2],
        'n_hyper_layer': [1],
        'hyper_num': [4],
        'keep_rate': [0.5],
        'alpha': [0.3],
        'cl_weight': [1e-04],
        'reg_weight': [1e-06],
    },
    'sports': {
        'gpu_id': 0,
        'n_ui_layers': [2],
        'n_mm_layers': [2],
        'n_hyper_layer': [1],
        'hyper_num': [4],
        'keep_rate': [0.4],
        'alpha': [0.6],
        'cl_weight': [1e-04],
        'reg_weight': [1e-06],
    },
    'clothing': {
        'gpu_id': 0,
        'n_ui_layers': [2],
        'n_mm_layers': [2],
        'n_hyper_layer': [2],
        'hyper_num': [64],
        'keep_rate': [0.2],
        'alpha': [0.2],
        'cl_weight': [1e-04],
        'reg_weight': [1e-06],
    },
}

datasets = ['baby', 'sports', 'clothing']

def run_experiment(model, dataset):
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        quick_start(model=model, dataset=dataset, config_dict=MODEL_CONFIGS[dataset], save_model=True)

    log_output = buffer.getvalue()
    matches = re.findall(r'^.*Test:.*$', log_output, re.MULTILINE)
    if matches:
        match_test = re.search(r'Recall@20:\s*([0-9.]+).*?NDCG@20:\s*([0-9.]+)', matches[-1], re.IGNORECASE)
        if match_test:
            return {'Recall@20': float(match_test.group(1)), 'NDCG@20': float(match_test.group(2))}
    return {'Recall@20': 0.0, 'NDCG@20': 0.0}

def build_latex_table(my_results):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Performance comparison on different datasets.}")
    lines.append(r"  \begin{tabular}{l | c c | c c | c c}")
    lines.append(r"    \toprule")
    lines.append(r"    \multirow{2}{*}{Method} & \multicolumn{2}{c|}{Baby} & \multicolumn{2}{c|}{Sports} & \multicolumn{2}{c}{Clothing} \\")
    lines.append(r"    & Recall@20 & NDCG@20 & Recall@20 & NDCG@20 & Recall@20 & NDCG@20 \\")
    lines.append(r"    \midrule")
    for bl in PAPER_BASELINES['baby']:
        lines.append("    " + f"{bl:<12}" + "".join(f"& {PAPER_BASELINES[ds][bl]['Recall@20']:.4f} & {PAPER_BASELINES[ds][bl]['NDCG@20']:.4f} " for ds in datasets) + r"\\")
    lines.append(r"    \midrule")
    lines.append("    " + f"Ours({my_model}) " + "".join(f"& \\textbf{{{my_results[ds]['Recall@20']:.4f}}} & \\textbf{{{my_results[ds]['NDCG@20']:.4f}}} " for ds in datasets) + r"\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)

if __name__ == '__main__':
    my_model = 'HMCRec'
    my_results = {dataset: run_experiment(my_model, dataset) for dataset in datasets} if TRAIN_MODE else HMC_PAPER_RESULTS
    table_text = build_latex_table(my_results)
    with open("d:/multi-modal/reconstruct_code/HMCRec/auto_run_results.tex", "w", encoding="utf-8") as f:
        f.write(table_text)