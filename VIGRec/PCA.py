import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

try:
    from svg_export import enable_editable_svg_text

    enable_editable_svg_text()
except Exception:
    pass


def load_interaction_matrix(inter_path, user_col=None, item_col=None, split_col=None, train_label=0):
    df = pd.read_csv(inter_path, sep='\t')
    if user_col is None:
        user_candidates = [c for c in df.columns if 'user' in c.lower()]
        if not user_candidates:
            raise ValueError('Cannot infer user column from interaction file.')
        user_col = user_candidates[0]
    if item_col is None:
        item_candidates = [c for c in df.columns if 'item' in c.lower()]
        if not item_candidates:
            raise ValueError('Cannot infer item column from interaction file.')
        item_col = item_candidates[0]
    if split_col is None:
        split_candidates = [c for c in df.columns if 'split' in c.lower() or 'label' in c.lower()]
        if not split_candidates:
            raise ValueError('Cannot infer split column from interaction file.')
        split_col = split_candidates[0]

    train_df = df[df[split_col] == train_label]
    n_users = int(df[user_col].max()) + 1
    n_items = int(df[item_col].max()) + 1
    rows = train_df[user_col].to_numpy(dtype=np.int64)
    cols = train_df[item_col].to_numpy(dtype=np.int64)
    data = np.ones(len(train_df), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items)), df, user_col, item_col, split_col


def load_fake_matrix(fake_path, shape):
    fake = sparse.load_npz(fake_path)
    if fake.shape[1] != shape[1]:
        raise ValueError(f'Fake matrix item dimension {fake.shape[1]} does not match real matrix {shape[1]}.')
    if fake.shape[0] == shape[0]:
        return fake
    if fake.shape[0] < shape[0]:
        pad = sparse.csr_matrix((shape[0] - fake.shape[0], shape[1]), dtype=fake.dtype)
        return sparse.vstack([fake, pad], format='csr')
    return fake[:shape[0], :]


def fit_pca_projection(*matrices, n_components=2, random_state=42):
    combined = np.vstack(matrices)
    projector = PCA(n_components=n_components, random_state=random_state)
    projector.fit(combined)
    return projector


def transform_pca(projector, matrix):
    return projector.transform(matrix)


def choose_target_users(train_matrix, target_item_ids, min_interactions=2):
    user_activity = np.asarray(train_matrix.sum(axis=1)).ravel()
    target_item_mask = np.asarray(train_matrix[:, target_item_ids].sum(axis=1)).ravel() > 0
    return np.where((user_activity >= min_interactions) & target_item_mask)[0]


def sample_virtual_users(train_matrix, n_samples, random_state=42):
    rng = np.random.default_rng(random_state)
    candidates = np.arange(train_matrix.shape[0])
    if len(candidates) <= n_samples:
        return candidates
    return rng.choice(candidates, size=n_samples, replace=False)


def plot_before_after(before_target, before_virtual, after_target, after_virtual, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(before_target[:, 0], before_target[:, 1], c='red', s=14, alpha=0.55, label='Target user')
    ax1.scatter(before_virtual[:, 0], before_virtual[:, 1], c='navy', s=16, alpha=0.85, label='Virtual user')
    ax1.set_title('Before')
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')
    ax1.grid(True, linestyle='--', alpha=0.35)
    ax1.legend(loc='best', framealpha=0.9)

    ax2.scatter(after_target[:, 0], after_target[:, 1], c='red', s=14, alpha=0.55, label='Target user')
    ax2.scatter(after_virtual[:, 0], after_virtual[:, 1], c='navy', s=16, alpha=0.85, label='Virtual user')
    ax2.set_title('After')
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    ax2.grid(True, linestyle='--', alpha=0.35)
    ax2.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real PCA visualization for VIGRec')
    parser.add_argument('--data-dir', type=str, default=r'd:\multi-modal\reconstruct_code\VIGRec\code\data\baby')
    parser.add_argument('--inter-file', type=str, default='baby.inter')
    parser.add_argument('--fake-data', type=str, default=None, help='Optional fake interaction matrix saved as .npz')
    parser.add_argument('--target-items', type=str, default=None, help='Comma-separated target item ids')
    parser.add_argument('--out', type=str, default='PCA.svg')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    inter_path = Path(args.data_dir) / args.inter_file
    train_matrix, df, user_col, item_col, split_col = load_interaction_matrix(inter_path)

    if args.target_items is not None:
        target_items = np.asarray([int(x) for x in args.target_items.split(',') if x.strip() != ''], dtype=np.int64)
    else:
        target_items = np.asarray([int(x) for x in sorted(df[item_col].unique())[:5]], dtype=np.int64)

    target_users = choose_target_users(train_matrix, target_items)
    if len(target_users) == 0:
        target_users = np.arange(min(100, train_matrix.shape[0]))

    virtual_users = sample_virtual_users(train_matrix, max(len(target_users), 100), random_state=args.seed)

    if args.fake_data is not None:
        fake_path = Path(args.fake_data)
        fake_matrix = load_fake_matrix(fake_path, train_matrix.shape)
        after_matrix = train_matrix + fake_matrix
    else:
        after_matrix = train_matrix.copy()

    before_target_matrix = train_matrix[target_users].toarray()
    before_virtual_matrix = train_matrix[virtual_users].toarray()
    after_target_matrix = after_matrix[target_users].toarray()
    after_virtual_matrix = after_matrix[virtual_users].toarray()

    projector = fit_pca_projection(
        before_target_matrix,
        before_virtual_matrix,
        after_target_matrix,
        after_virtual_matrix,
        n_components=2,
        random_state=args.seed,
    )

    before_target = transform_pca(projector, before_target_matrix)
    before_virtual = transform_pca(projector, before_virtual_matrix)
    after_target = transform_pca(projector, after_target_matrix)
    after_virtual = transform_pca(projector, after_virtual_matrix)

    plot_before_after(before_target, before_virtual, after_target, after_virtual, args.out)
