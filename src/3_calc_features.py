# Feature Engineering: Exact Density, Curvature, and Projection Features
import os
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import config

# ---------------- CONFIGURATION ----------------
EPS = 1e-6
DTYPE = np.float32
N_WORKERS = max(1, mp.cpu_count() - 1)

# --- Vertical Projection Parameters ---
TAU_AMPL = 0.12
TAU_SIGN = 0.08
TAU_BLEND = 0.18
GAMMA = 1.2
EDGE_RADIUS_MUL = 2.2
EDGE_ALPHA_MAX = 0.55
EDGE_ITERS = 2

# --- Curvature Parameters ---
CURV_NEI_RADIUS = 2.0
CURV_MIN_NEI = 12
BOUND_MAX_DIST = 1.2
DECAY_SIGMA = 0.4
R_CAP_MIN = 0.10
R_CAP_MAX = 50.0

# --- Density Calculation (Block Size) ---
DENSITY_CHUNK = 4096

# --- Boundary Node Identification ---
BOUND_K = 5
BOUND_RADIUS = 1.0

TARGET_COLUMNS = ["SigxxE", "SigyyE", "Sigxy", "Epsxx", "Epsyy", "Gamxy", "Utot"]


# ======================= Utilities =======================
def _estimate_grid_step(nodes_xy):
    """Estimates the median grid step size."""
    if len(nodes_xy) < 2: return 1.0
    d, _ = KDTree(nodes_xy).query(nodes_xy, k=2)
    return float(np.median(d[:, 1]) + 1e-6)


def _split_connected(points_xy, eps):
    """Splits a set of points into connected components."""
    if len(points_xy) == 0: return []
    tree = KDTree(points_xy)
    neigh = tree.query_ball_point(points_xy, r=eps)
    N = len(points_xy)
    seen = np.zeros(N, bool)
    out = []
    for i in range(N):
        if seen[i]: continue
        st = [i]
        seen[i] = True
        comp = []
        while st:
            j = st.pop()
            comp.append(j)
            for k in neigh[j]:
                if not seen[k]:
                    seen[k] = True
                    st.append(k)
        out.append(points_xy[np.array(comp)])
    return out


# ======================= Feature Calculation =======================
def calculate_density(nodes, excavated_nodes, chunk=DENSITY_CHUNK):
    """
    Calculates exact density using block matrix operations.
    Formula: 1 / mean(dist(node_i, all_excavated_nodes_j))
    """
    N = len(nodes)
    if not excavated_nodes.size or N == 0:
        return np.zeros(N, dtype=DTYPE)

    X = nodes[:, :2].astype(np.float64, copy=False)
    E = excavated_nodes[:, :2].astype(np.float64, copy=False)

    N2 = (X * X).sum(axis=1, keepdims=True)
    sum_d = np.zeros(N, dtype=np.float64)
    M = len(E)

    for s in range(0, M, chunk):
        F = E[s:s + chunk]
        F2 = (F * F).sum(axis=1, keepdims=True).T
        G = X @ F.T
        d2 = N2 + F2 - 2.0 * G
        np.maximum(d2, 0.0, out=d2)
        d = np.sqrt(d2, dtype=np.float64)
        sum_d += d.sum(axis=1)

    avg = sum_d / (M + 1e-12)
    out = np.where(avg > 1e-6, 1.0 / (avg + 1e-6), 0.0)
    return out.astype(DTYPE)


def _kasa_circle_fit(P):
    """Fits a circle to points using Kasa's method."""
    x = P[:, 0]
    y = P[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, c = sol
        R2 = cx * cx + cy * cy + c
        if not np.isfinite(R2) or R2 <= 0: return None
        R = float(np.sqrt(R2))
        return float(np.clip(R, R_CAP_MIN, R_CAP_MAX))
    except np.linalg.LinAlgError:
        return None


def _pca_cornerness(P):
    """Estimates 'cornerness' using PCA eigenvalues."""
    Q = P - P.mean(axis=0, keepdims=True)
    C = (Q.T @ Q) / max(len(P) - 1, 1)
    evals, _ = np.linalg.eigh(C)
    if evals.size == 1:
        l1 = float(evals[0])
        l2 = 0.0
    else:
        l1 = float(evals[-1])
        l2 = float(evals[-2])
    s = l1 + l2
    return 0.0 if s <= 1e-12 else float(np.clip(l2 / s, 0.0, 1.0))


def _precompute_boundary_curvatures(B, kdB):
    M = len(B)
    kappa = np.zeros(M, np.float64)
    for bi in range(M):
        ids = kdB.query_ball_point(B[bi], r=CURV_NEI_RADIUS)
        if len(ids) < CURV_MIN_NEI: continue
        local = B[np.asarray(ids)]
        corner = _pca_cornerness(local)
        if corner <= 1e-6: continue
        R = _kasa_circle_fit(local)
        if R is None or R <= 0: continue
        kappa[bi] = corner * (1.0 / R)
    m = float(kappa.max())
    return (kappa * (100.0 / m)).astype(np.float64) if m > 0 else kappa


def calculate_curvature_fast(nodes, boundary_nodes):
    """Calculates curvature field propagated from boundary nodes."""
    N = len(nodes)
    if not boundary_nodes.size or N == 0: return np.zeros(N, DTYPE)
    X = nodes[:, :2].astype(np.float64)
    B = boundary_nodes[:, :2].astype(np.float64)
    kdB = KDTree(B)
    d, idx = kdB.query(X, k=1)
    mask = d <= BOUND_MAX_DIST
    if not np.any(mask): return np.zeros(N, DTYPE)

    kappa_b = _precompute_boundary_curvatures(B, kdB)
    feat = np.zeros(N, np.float64)
    if kappa_b.max() > 0:
        w = np.exp(-d[mask] / max(DECAY_SIGMA, 1e-9))
        feat[mask] = kappa_b[idx[mask]] * w
    return feat.astype(DTYPE)


def calculate_signed_dist_norm(nodes, exc_mask, boundary_nodes, mean_width):
    """Calculates normalized signed distance from the excavation boundary."""
    N = len(nodes)
    out = np.zeros(N, DTYPE)
    if not boundary_nodes.size or N == 0: return out
    kdB = KDTree(boundary_nodes[:, :2].astype(np.float64))
    d, _ = kdB.query(nodes[:, :2].astype(np.float64), k=1)
    sign = np.where(exc_mask, -1.0, +1.0).astype(np.float64)
    denom = np.maximum(mean_width.astype(np.float64), EPS)
    return (sign * (d / denom)).astype(DTYPE)


def _edge_aware_smooth_fast(nodes_xy, values, blend_mask, radius, alpha_max=0.55, iters=2):
    """Applies edge-aware smoothing to feature fields."""
    X = nodes_xy.astype(np.float64)
    vals = values.astype(np.float64).copy()
    m = np.clip(blend_mask.astype(np.float64), 0, 1)
    kd = KDTree(X)
    neigh = kd.query_ball_point(X, r=radius)
    rows, cols, data = [], [], []
    for i, ids in enumerate(neigh):
        if not ids: continue
        nbr = np.asarray(ids, int)
        d = np.linalg.norm(X[nbr] - X[i], axis=1) + 1e-9
        w = np.exp(-(d * d) / (2.0 * (0.35 * radius) ** 2))
        w /= max(w.sum(), 1e-12)
        rows.extend([i] * len(nbr))
        cols.extend(nbr.tolist())
        data.extend(w.tolist())

    N = len(X)
    if not data: return vals.astype(DTYPE)
    W = csr_matrix((data, (rows, cols)), shape=(N, N))

    rough = np.abs(vals - W.dot(vals))
    if np.any(rough > 0):
        q75, q90 = np.percentile(rough, [75, 90])
    else:
        q75, q90 = 0.0, 1.0
    scale = max(q90 - q75, 1e-6)
    sharp = np.clip((rough - q75) / scale, 0, 1)
    alpha = alpha_max * m * sharp

    for _ in range(max(0, int(iters))):
        nbr_mean = W.dot(vals)
        vals = (1.0 - alpha) * vals + alpha * nbr_mean
    return vals.astype(DTYPE)


def calculate_vertical_projection(nodes, excavated_nodes):
    """Calculates the vertical projection influence ('hourglass' effect)."""
    N = len(nodes)
    if not excavated_nodes.size or N == 0: return np.zeros(N, DTYPE)
    X = nodes[:, :2].astype(np.float64)
    E = excavated_nodes[:, :2].astype(np.float64)
    h = _estimate_grid_step(X)
    comps = _split_connected(E, eps=1.5 * h)

    if len(comps) == 0: return np.zeros(N, DTYPE)
    trees = [KDTree(c) for c in comps]
    M = len(comps)
    r = np.zeros((N, M))
    dy = np.zeros((N, M))

    for j, (c, t) in enumerate(zip(comps, trees)):
        d, idx = t.query(X, k=1)
        r[:, j] = d + 1e-6
        near = c[idx]
        dy[:, j] = X[:, 1] - near[:, 1]

    p = dy / r
    abs_p = np.abs(p) ** GAMMA
    pos = (p > 0)
    neg = (p < 0)

    def soft(mask):
        a = np.where(mask, abs_p, 0.0)
        w = np.where(mask, np.exp((abs_p + 1e-12) / TAU_AMPL), 0.0)
        s = w.sum(axis=1, keepdims=True) + 1e-12
        return np.clip((w * a).sum(axis=1) / s.ravel(), 0, 1)

    A_pos = soft(pos)
    A_neg = soft(neg)
    dA = A_pos - A_neg
    sign = np.tanh(dA / TAU_SIGN)
    A_dom = np.maximum(A_pos, A_neg)
    A_mean = np.sqrt(0.5 * (A_pos * A_pos + A_neg * A_neg))
    beta = np.exp(-(dA * dA) / (2.0 * TAU_BLEND * TAU_BLEND))

    vp0 = np.clip(((1.0 - beta) * A_dom + beta * A_mean) * sign, -1, 1).astype(DTYPE)
    vp = _edge_aware_smooth_fast(X, vp0, beta, radius=EDGE_RADIUS_MUL * h,
                                 alpha_max=EDGE_ALPHA_MAX, iters=EDGE_ITERS)
    return np.clip(vp, -1, 1).astype(DTYPE)


# ================= Global Pipeline =================
def build_global_params(df: pd.DataFrame) -> pd.DataFrame:
    """Computes global geometric ratios and parameters."""
    req = ["distance", "vertical_shift", "width_tunnel1", "height_tunnel1", "width_tunnel2", "height_tunnel2"]
    for c in req:
        if c not in df.columns: raise ValueError(f"Missing required column '{c}'")

    w1 = df["width_tunnel1"].astype(np.float64).values
    h1 = df["height_tunnel1"].astype(np.float64).values
    w2 = df["width_tunnel2"].astype(np.float64).values
    h2 = df["height_tunnel2"].astype(np.float64).values
    dist = df["distance"].astype(np.float64).values
    vsh = df["vertical_shift"].astype(np.float64).values

    mean_width = (w1 + w2) / 2.0
    mean_height = (h1 + h2) / 2.0
    aspect1 = np.where(np.abs(h1) > EPS, w1 / (h1 + EPS), np.nan)
    aspect2 = np.where(np.abs(h2) > EPS, w2 / (h2 + EPS), np.nan)
    dist_norm = np.where(np.abs(mean_width) > EPS, dist / (mean_width + EPS), np.nan)
    shift_norm = np.where(np.abs(mean_height) > EPS, vsh / (mean_height + EPS), np.nan)
    area_ratio = (w1 * h1) / (w2 * h2 + EPS)

    out = df.copy()
    out["mean_width"] = mean_width.astype(DTYPE)
    out["mean_height"] = mean_height.astype(DTYPE)
    out["aspect1"] = aspect1.astype(DTYPE)
    out["aspect2"] = aspect2.astype(DTYPE)
    out["dist_norm"] = dist_norm.astype(DTYPE)
    out["shift_norm"] = shift_norm.astype(DTYPE)
    out["area_ratio"] = area_ratio.astype(DTYPE)
    out = out.drop(columns=req, errors="ignore")
    return out


def _mark_bound_exc(df, nodes, exc_mask, non_exc_nodes):
    """Marks boundary nodes in the soil (Bound_Exc = 1)."""
    if "Bound_Exc" not in df.columns:
        df["Bound_Exc"] = 0
    if non_exc_nodes.size == 0 or not np.any(exc_mask):
        return np.array([])
    kdt_non = KDTree(non_exc_nodes)
    k_query = min(BOUND_K, len(non_exc_nodes))
    dist, idx = kdt_non.query(nodes[exc_mask], k=k_query, distance_upper_bound=BOUND_RADIUS)
    valid = idx[idx < len(non_exc_nodes)]
    if valid.size == 0:
        return np.array([])
    non_exc_idx = df.index[~exc_mask][np.unique(valid)]
    df.loc[non_exc_idx, "Bound_Exc"] = 1
    return non_exc_nodes[np.unique(valid)]


def _calc_for_one_file(input_path, output_path):
    """Processes a single simulation CSV file."""
    df = pd.read_csv(input_path)
    base_required = ["X", "Y", "distance", "vertical_shift", "width_tunnel1", "height_tunnel1", "width_tunnel2",
                     "height_tunnel2"]
    if not all(c in df.columns for c in base_required):
        print(f"  Skipping {os.path.basename(input_path)}: missing required columns.")
        return False

    # Global Parameters
    df = build_global_params(df)
    if not all(c in df.columns for c in ["X", "Y"]):
        print(f"  Skipping {os.path.basename(input_path)}: missing spatial coordinates.")
        return False

    nodes = df[["X", "Y"]].values.astype(np.float64)
    mean_width = df["mean_width"].astype(np.float64).values

    # Excavation status
    if "Excavated_soil" not in df.columns:
        df["Excavated_soil"] = 0
    exc_mask = df["Excavated_soil"].astype(int).values == 1
    exc_nodes = nodes[exc_mask]
    non_exc_nodes = nodes[~exc_mask]

    # Boundary Identification
    boundary_nodes = _mark_bound_exc(df, nodes, exc_mask, non_exc_nodes)

    # Feature Computation
    df["Vertical_Projection"] = calculate_vertical_projection(nodes, exc_nodes)
    df["Signed_Dist_Norm"] = calculate_signed_dist_norm(nodes, exc_mask, boundary_nodes, mean_width) \
        if boundary_nodes.size > 0 else np.zeros(len(df), dtype=DTYPE)
    df["Curvature"] = calculate_curvature_fast(nodes, boundary_nodes)
    df["Density_Excavated_Distances"] = calculate_density(nodes, exc_nodes)

    # Overlap Index
    df["Overlap_Index"] = 0.0
    if exc_nodes.size > 0:
        X = nodes[:, :2].astype(np.float64)
        E = exc_nodes[:, :2].astype(np.float64)
        h = _estimate_grid_step(X)
        comps = _split_connected(E, eps=1.5 * h)
        trees = [KDTree(c) for c in comps]
        if len(trees) >= 2:
            D = []
            for t in trees:
                d1, _ = t.query(X, k=1)
                D.append(d1)
            D = np.vstack(D).T
            D_sorted = np.sort(D, axis=1)
            m1 = D_sorted[:, 0]
            m2 = D_sorted[:, 1]
            L = np.maximum(mean_width, EPS)
            df["Overlap_Index"] = (np.exp(-m1 / L) * np.exp(-m2 / L)).astype(DTYPE)

    # Formatting and Cleanup
    df = df.drop(columns=[c for c in ["NodeID", "Boundary_Proximity", "Distance_to_Boundary"] if c in df.columns],
                 errors="ignore")

    global_cols = ["X", "Y", "mean_width", "mean_height", "aspect1", "aspect2", "dist_norm", "shift_norm", "area_ratio"]
    feature_cols = ["Vertical_Projection", "Signed_Dist_Norm", "Curvature", "Density_Excavated_Distances",
                    "Overlap_Index"]
    target_cols = [c for c in TARGET_COLUMNS if c in df.columns]
    service_cols = [c for c in ["Excavated_soil", "Bound_Exc"] if c in df.columns]

    ordered = [c for c in global_cols if c in df.columns] + \
              [c for c in feature_cols if c in df.columns] + \
              target_cols + service_cols
    other = [c for c in df.columns if c not in ordered]
    df = df[ordered + other]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return True


def main():
    input_dir = config.raw_data_dir
    output_dir = config.features_dir
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return

    files = [f for f in sorted(os.listdir(input_dir)) if f.endswith(".csv")]
    if not files:
        print("No CSV input files found.")
        return

    print(f"CPU Cores: {mp.cpu_count()} -> Active Workers: {N_WORKERS}")
    futures = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for filename in files:
            inp = os.path.join(input_dir, filename)
            out = os.path.join(output_dir, filename)
            futures.append(ex.submit(_calc_for_one_file, inp, out))
        ok = 0
        for fut in as_completed(futures):
            try:
                ok += int(bool(fut.result()))
            except Exception as e:
                print("Worker Execution Error:", e)

    print(f"\nFeature generation completed. Successfully processed: {ok}/{len(files)}")


if __name__ == "__main__":
    main()