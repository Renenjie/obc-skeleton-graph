# -*- coding: utf-8 -*-
"""
evaluate_sensitivity_results.py
遍历 A_para_analysis 下由 run_sensitivity_analysis.py 生成的矩阵目录，
与 GT 对比计算指标，结果写入 para_analysis_result.txt。
【修改】增加平均节点数统计（含 GT 节点数对照）
"""
import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import entropy

# --- 定位仓库 code/ 目录（含 paths.py），保证脚本可在任意工作目录下运行 ---
_CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").is_file())
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from paths import RESULT_ROOT, DEFAULT_DATASET, dataset_paths, env_path, print_config

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
np.seterr(invalid='ignore')

# ======================== 配置（相对仓库根目录，见 code/paths.py） ======================== #
# 与 Parameter_sensitivity_analysis.py 的输出目录保持一致；
# 如需评估其它位置的旧结果，可设置环境变量 SKELETON_ANALYSIS_DIR
ANALYSIS_DIR = env_path("SKELETON_ANALYSIS_DIR",
                        RESULT_ROOT / "parameter_sensitivity_analysis")
# 数据集设置：只改下一行（相对 <repo>/data，无需绝对路径）；DEFAULT_DATASET = 跟随全局
DATASET = DEFAULT_DATASET                          # 例："skeleton_756"
_, GT_DIR, DATASET_TAG = dataset_paths(DATASET)    # <repo>/data/Matrix_<N>
RESULT_FILE = ANALYSIS_DIR / f"para_analysis_result_{DATASET_TAG}.txt"

DIR_PREFIX = f"A_matrix_{DATASET_TAG}_t5_"

ADJ_SUFFIX = "adj.npy"
NODES_SUFFIX = "nodes.npy"
CYCLE_MAX_LEN = 8
CYCLE_MAX_COUNT = 10000

PARAM_ORDER = [
    'min_branch_len',
    'track_steps',
    'circular_min_distance',
    'merge_dist',
    'min_chord_length',
    'min_spacing',
    'min_edge_len',
    'temp_min_edge_len'
]

# ======================== 评估函数（与原版一致） ======================== #
def find_file_safe(directory, stem, suffix):
    path = directory / f"{stem}{suffix}"
    if path.exists():
        return path
    stem_lower = stem.lower()
    for f in directory.glob("*" + suffix):
        if f.stem == stem:
            return f
    for f in directory.glob("*" + suffix):
        if f.stem.lower() == stem_lower:
            return f
    stem_strip = stem.strip()
    if stem_strip != stem:
        path = directory / f"{stem_strip}{suffix}"
        if path.exists():
            return path
        stem_strip_lower = stem_strip.lower()
        for f in directory.glob("*" + suffix):
            if f.stem.strip() == stem_strip:
                return f
            if f.stem.strip().lower() == stem_strip_lower:
                return f
    return None

def load_graph_from_adj_nodes(adj_path, nodes_path):
    adj = np.load(adj_path)
    nodes = np.load(nodes_path)
    if adj.size == 0:
        return nx.Graph()
    adj = (adj + adj.T) > 0.0
    G = nx.from_numpy_array(adj.astype(int))
    for i, coord in enumerate(nodes):
        G.nodes[i]['coord'] = coord
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def load_gt_csv(csv_path):
    adj = pd.read_csv(csv_path, header=None).values
    if adj.size == 0:
        return nx.Graph()
    adj = (adj + adj.T) > 0
    G = nx.from_numpy_array(adj.astype(int))
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def degree_distribution_js_distance(G1, G2):
    n1 = G1.number_of_nodes()
    n2 = G2.number_of_nodes()
    if n1 == 0 and n2 == 0:
        return 0.0
    if n1 == 0 or n2 == 0:
        return np.nan
    deg_seq1 = np.array([d for _, d in G1.degree()])
    deg_seq2 = np.array([d for _, d in G2.degree()])
    max_deg = max(deg_seq1.max(), deg_seq2.max())
    counts1 = np.bincount(deg_seq1, minlength=max_deg + 1).astype(np.float64)
    counts2 = np.bincount(deg_seq2, minlength=max_deg + 1).astype(np.float64)
    epsilon = 1e-12
    k = len(counts1)
    p = (counts1 + epsilon) / (counts1.sum() + epsilon * k)
    q = (counts2 + epsilon) / (counts2.sum() + epsilon * k)
    m = (p + q) / 2.0
    return (entropy(p, m) + entropy(q, m)) / 2.0

def spectral_distance(G1, G2, bins=10):
    n1, n2 = G1.number_of_nodes(), G2.number_of_nodes()
    if n1 == 0 and n2 == 0:
        return 0.0
    if n1 == 0 or n2 == 0:
        return np.nan
    if n1 < 2 or n2 < 2:
        return np.nan
    try:
        L1 = nx.normalized_laplacian_matrix(G1).toarray()
        L2 = nx.normalized_laplacian_matrix(G2).toarray()
    except Exception:
        return np.nan
    L1 = (L1 + L1.T) / 2.0
    L2 = (L2 + L2.T) / 2.0
    eig1 = np.linalg.eigvalsh(L1)
    eig2 = np.linalg.eigvalsh(L2)
    eig1 = np.clip(eig1, 0, 2)
    eig2 = np.clip(eig2, 0, 2)
    num_bins = min(bins, n1, n2)
    hist1, _ = np.histogram(eig1, bins=num_bins, range=(0, 2))
    hist2, _ = np.histogram(eig2, bins=num_bins, range=(0, 2))
    hist1 = hist1.astype(np.float64) + 1e-12
    hist2 = hist2.astype(np.float64) + 1e-12
    hist1 /= hist1.sum()
    hist2 /= hist2.sum()
    M = (hist1 + hist2) / 2
    return (entropy(hist1, M) + entropy(hist2, M)) / 2

def cyclomatic_number(G):
    if G.number_of_nodes() == 0:
        return 0
    return G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G)

def degree_assortativity(G):
    if G.number_of_nodes() < 2:
        return 0.0
    try:
        return nx.degree_assortativity_coefficient(G)
    except Exception:
        return 0.0

def components_count_error(G, G_gt):
    return abs(nx.number_connected_components(G) - nx.number_connected_components(G_gt))

def density_error(G, G_gt):
    return abs(nx.density(G) - nx.density(G_gt))

def algebraic_connectivity_error(G, G_gt):
    n1 = G.number_of_nodes()
    n2 = G_gt.number_of_nodes()
    if n1 < 2 or n2 < 2:
        return np.nan
    try:
        L1 = nx.laplacian_matrix(G).toarray()
        L2 = nx.laplacian_matrix(G_gt).toarray()
    except Exception:
        return np.nan
    eigs1 = np.linalg.eigvalsh(L1)
    eigs2 = np.linalg.eigvalsh(L2)
    alg_conn1 = eigs1[1] if len(eigs1) > 1 else 0.0
    alg_conn2 = eigs2[1] if len(eigs2) > 1 else 0.0
    return abs(alg_conn1 - alg_conn2)

def cycle_length_distribution_js_distance(G1, G2, max_cycle_len=CYCLE_MAX_LEN, max_count=CYCLE_MAX_COUNT):
    def get_cycle_lengths(G):
        if G.number_of_nodes() == 0:
            return np.array([])
        cycles_set = set()
        count = 0
        try:
            for cycle in nx.simple_cycles(G, length_bound=max_cycle_len):
                if len(cycle) >= 3:
                    normalized = tuple(cycle)
                    min_idx = np.argmin(normalized)
                    rotated = normalized[min_idx:] + normalized[:min_idx]
                    rev_rotated = tuple(reversed(rotated))
                    standard = min(rotated, rev_rotated)
                    cycles_set.add(standard)
                    count += 1
                    if count > max_count:
                        return None
        except Exception:
            return None
        lengths = [len(c) for c in cycles_set]
        return np.array(lengths)

    len1 = get_cycle_lengths(G1)
    len2 = get_cycle_lengths(G2)
    if len1 is None or len2 is None:
        return np.nan
    if len(len1) == 0 and len(len2) == 0:
        return 0.0
    if len(len1) == 0 or len(len2) == 0:
        return 1.0
    bins = np.arange(3, max_cycle_len + 2)
    hist1, _ = np.histogram(len1, bins=bins)
    hist2, _ = np.histogram(len2, bins=bins)
    hist1 = hist1.astype(np.float64) + 1e-12
    hist2 = hist2.astype(np.float64) + 1e-12
    hist1 /= hist1.sum()
    hist2 /= hist2.sum()
    M = (hist1 + hist2) / 2
    return (entropy(hist1, M) + entropy(hist2, M)) / 2

# ======================== 【新增】评估目录，返回指标 + 节点数 ======================== #
def evaluate_directory(matrix_dir):
    """
    评估 matrix_dir 下所有 *nodes.npy 和 *adj.npy，与 GT 对比。
    返回字典：
      - 8 项指标均值
      - 'node_count_mean'：方法生成图的平均节点数
      - 'gt_node_count_mean'：对应 GT 的平均节点数（同样本对）
      - 'num_samples'：成功匹配的样本数
    """
    matrix_dir = Path(matrix_dir)
    if not matrix_dir.exists():
        return None

    gt_files = sorted(GT_DIR.glob("*.csv"))
    if not gt_files:
        print("错误：GT 目录中未找到任何 CSV 文件！")
        return None

    results = []
    for gt_path in gt_files:
        stem = gt_path.stem
        a_adj = find_file_safe(matrix_dir, stem, ADJ_SUFFIX)
        a_nodes = find_file_safe(matrix_dir, stem, NODES_SUFFIX)
        if not (a_adj and a_nodes):
            continue

        try:
            G_gt = load_gt_csv(gt_path)
            G_a = load_graph_from_adj_nodes(a_adj, a_nodes)
        except Exception as e:
            print(f"    加载失败 {stem}: {e}")
            continue

        js_dist = degree_distribution_js_distance(G_a, G_gt)
        spec_dist = spectral_distance(G_a, G_gt)
        cyclo_gt = cyclomatic_number(G_gt)
        cyclo_a = cyclomatic_number(G_a)
        cyclo_err = abs(cyclo_a - cyclo_gt)
        cycle_js = cycle_length_distribution_js_distance(G_a, G_gt)
        assort_gt = degree_assortativity(G_gt)
        assort_a = degree_assortativity(G_a)
        assort_err = abs(assort_a - assort_gt)
        comp_err = components_count_error(G_a, G_gt)
        dens_err = density_error(G_a, G_gt)
        alg_conn_err = algebraic_connectivity_error(G_a, G_gt)

        # 【新增】记录节点数
        results.append({
            'node_count': G_a.number_of_nodes(),
            'gt_node_count': G_gt.number_of_nodes(),
            'js_dist': js_dist,
            'spec_dist': spec_dist,
            'cyclo_err': cyclo_err,
            'cycle_js': cycle_js,
            'assort_err': assort_err,
            'comp_err': comp_err,
            'dens_err': dens_err,
            'alg_conn_err': alg_conn_err
        })

    if not results:
        return None

    df = pd.DataFrame(results)
    metrics = {}
    for col in df.columns:
        series = df[col].dropna()
        metrics[col] = series.mean() if len(series) > 0 else np.nan
    metrics['num_samples'] = len(df)
    return metrics

# ======================== 主程序 ======================== #
def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    print_config()
    print(f"分析目录: {ANALYSIS_DIR}")
    print(f"真值目录: {GT_DIR}")

    all_dirs = [d for d in ANALYSIS_DIR.iterdir() if d.is_dir() and d.name.startswith(DIR_PREFIX)]
    if not all_dirs:
        print(f"未在 {ANALYSIS_DIR} 找到以 {DIR_PREFIX} 开头的目录。")
        sys.exit(1)

    experiments = []
    for d in all_dirs:
        suffix = d.name[len(DIR_PREFIX):]
        if '=' not in suffix:
            continue
        param_name, value_str = suffix.split('=', 1)
        try:
            value = int(value_str)
        except ValueError:
            value = value_str
        experiments.append((param_name, value, d))

    from collections import defaultdict
    grouped = defaultdict(list)
    for param, val, d in experiments:
        grouped[param].append((val, d))

    ordered_params = []
    for p in PARAM_ORDER:
        if p in grouped:
            ordered_params.append(p)
    for p in sorted(grouped.keys()):
        if p not in ordered_params:
            ordered_params.append(p)

    with open(RESULT_FILE, 'w', encoding='utf-8') as f_out:
        for param in ordered_params:
            items = sorted(grouped[param], key=lambda x: (isinstance(x[0], str), x[0]))
            for val, d in items:
                title = f"{param}={val}"
                print(f"\n处理: {title}  ->  {d.name}")
                metrics = evaluate_directory(d)
                f_out.write(f"{title}\n")

                if metrics:
                    # 【新增】先输出节点数
                    node_mean = metrics.get('node_count', np.nan)
                    gt_mean = metrics.get('gt_node_count', np.nan)
                    n_samples = metrics.get('num_samples', 0)
                    if not np.isnan(node_mean):
                        f_out.write(f"node_count: {node_mean:.4f}\n")
                    else:
                        f_out.write(f"node_count: N/A\n")
                    if not np.isnan(gt_mean):
                        f_out.write(f"gt_node_count: {gt_mean:.4f}\n")

                    # 原有 8 项指标
                    for key in ['js_dist', 'spec_dist', 'cyclo_err', 'cycle_js',
                                'assort_err', 'comp_err', 'dens_err', 'alg_conn_err']:
                        if key in metrics and not np.isnan(metrics[key]):
                            f_out.write(f"{key}: {metrics[key]:.6f}\n")
                        else:
                            f_out.write(f"{key}: N/A\n")
                    f_out.write(f"num_samples: {n_samples}\n")
                else:
                    f_out.write("无有效结果\n")
                f_out.write("\n")
                f_out.flush()

    print(f"\n所有评估完成，结果已保存至: {RESULT_FILE}")

if __name__ == "__main__":
    main()