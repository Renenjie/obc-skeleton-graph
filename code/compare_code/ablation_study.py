import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import entropy
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
np.seterr(invalid='ignore')


# ==================== 路径配置（以 file/ 文件夹为根，全部相对拼接） ==================== #
def find_file_root(script_path: Path) -> Path:
    """从脚本位置向上找 `file/` 根目录：
    优先找含 `data/` 子目录的那一级；找不到时兜底取上三级。
    """
    start = script_path.resolve().parent          # 脚本所在目录
    for p in [start] + list(start.parents):
        if (p / "data").is_dir():
            return p
    return start.parents[2]                        # 兜底：code/compare_code -> file


FILE_ROOT   = find_file_root(Path(__file__))       # 即 E:/skeleton_v1/file
DATA_DIR    = FILE_ROOT / "data"                   # E:/skeleton_v1/file/data
RESULT_ROOT = FILE_ROOT / "Experiment_results"     # E:/skeleton_v1/file/Experiment_results

# 数据集（只改这一行）；tag 用于拼接 Matrix_<N> / A_matrix_<N>
DATASET     = "skeleton_3780"
DATASET_TAG = DATASET.split("_")[-1]               # "skeleton_3780" -> "3780"

# 相对 FILE_ROOT 拼接出的两个目录
gt_dir = DATA_DIR / f"Matrix_{DATASET_TAG}"        # <file>/data/Matrix_3780
a_dir  = RESULT_ROOT / f"A_matrix_{DATASET_TAG}_t4"   # <file>/Experiment_results/A_matrix_3780
# ================================================================================== #


def extract_method_label(dir_path):
    """从目录路径中提取方法名，如 A_matrix_3780_t1 → A"""
    return Path(dir_path).name.split('_')[0]


method_a = extract_method_label(a_dir)

print(f"根目录 (file): {FILE_ROOT}")
print(f"数据集:        {DATASET}  (GT: {gt_dir})")
print(f"消融评估方法:  {method_a} vs GT")
print(f"评估目录:      {a_dir}")

for d, name in [(a_dir, method_a), (gt_dir, "GT")]:
    if not d.exists():
        print(f"❌ 错误: {name} 目录不存在: {d}")
        sys.exit(1)

ADJ_SUFFIX = "adj.npy"
NODES_SUFFIX = "nodes.npy"
CYCLE_MAX_LEN = 8
CYCLE_MAX_COUNT = 10000

# ======================== 函数定义 ========================
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

# ======================== 数据收集 ========================
gt_files = sorted(gt_dir.glob("*.csv"))
print(f"GT 文件总数: {len(gt_files)}")
if not gt_files:
    print("错误：GT 目录中未找到任何 CSV 文件！")
    sys.exit(1)

missing_stems = []
results = []

try:
    from tqdm import tqdm
    iterator = tqdm(gt_files, desc="处理进度")
except ImportError:
    iterator = gt_files

for gt_path in iterator:
    stem = gt_path.stem
    a_adj = find_file_safe(a_dir, stem, ADJ_SUFFIX)
    a_nodes = find_file_safe(a_dir, stem, NODES_SUFFIX)

    if not (a_adj and a_nodes):
        missing_stems.append(stem)
        continue

    G_gt = load_gt_csv(gt_path)
    G_a = load_graph_from_adj_nodes(a_adj, a_nodes)

    gt_n = G_gt.number_of_nodes()
    n_a = G_a.number_of_nodes()

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

    results.append({
        'filename': stem,
        'gt_node_count': gt_n,
        'node_count_A': n_a,
        'js_dist': js_dist,
        'spec_dist': spec_dist,
        'gt_cyclo': cyclo_gt,
        'cyclo_err': cyclo_err,
        'cycle_js': cycle_js,
        'gt_assort': assort_gt,
        'assort_err': assort_err,
        'comp_err': comp_err,
        'dens_err': dens_err,
        'alg_conn_err': alg_conn_err
    })

df = pd.DataFrame(results)
print(f"成功处理样本数: {len(df)}")
print(f"缺失文件样本数: {len(missing_stems)}")
if missing_stems:
    print("缺失文件列表（前20个）:", missing_stems[:20])

if df.empty:
    print("\n严重警告：没有任何成功匹配的样本。")
    sys.exit(1)

# ======================== 空图统计 ========================
empty_A = df[df['node_count_A'] == 0]
print(f"\n方法 {method_a} 空图: {len(empty_A)}")

# ======================== 指标汇总输出（按指定顺序） ========================
print("\n" + "="*70)
print("                    消融实验结果汇总（A vs GT）")
print("="*70)

metrics = [
    ("节点数", "node_count_A", "与GT相近为佳"),
    ("度分布JS散度", "js_dist", "↓"),
    ("边密度误差", "dens_err", "↓"),
    ("度相关性绝对误差", "assort_err", "↓"),
    ("循环数绝对误差", "cyclo_err", "↓"),
    ("环长度分布JS散度", "cycle_js", "↓"),
    ("连通分量数量误差", "comp_err", "↓"),
    ("代数连通度误差", "alg_conn_err", "↓"),
    ("谱距离", "spec_dist", "↓")
]

print(f"\n{'指标':<25}{'均值':<12}{'有效样本数'}")
print("-" * 50)
for name, col, _ in metrics:
    series = df[col].dropna()
    if len(series) == 0:
        print(f"{name:<25}{'N/A':<12}0")
    else:
        print(f"{name:<25}{series.mean():<12.4f}{len(series)}")

print("\n所有计算完成。")