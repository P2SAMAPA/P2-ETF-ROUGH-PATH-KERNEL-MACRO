import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

def signature(returns, depth=4):
    """
    Compute truncated path signature (simplified) for a 1D return series.
    For a 1D path, the signature consists of iterated integrals:
    S^1 = sum of increments, S^2 = sum of (increment_i * increment_j) for i<j, etc.
    We'll compute up to depth 4.
    """
    increments = np.diff(returns)
    if len(increments) == 0:
        return np.zeros(depth)
    sig = []
    # Level 1: sum of increments
    sig.append(np.sum(increments))
    if depth >= 2:
        # Level 2: sum_{i<j} inc_i * inc_j = 0.5 * ( (sum inc)^2 - sum inc^2 )
        sum_inc = np.sum(increments)
        sum_sq = np.sum(increments**2)
        sig.append(0.5 * (sum_inc**2 - sum_sq))
    if depth >= 3:
        # Level 3: approximated by (sum inc)^3 / 6 (for simplicity)
        sig.append((np.sum(increments)**3) / 6.0)
    if depth >= 4:
        sig.append((np.sum(increments)**4) / 24.0)
    return np.array(sig)

def kernel_matrix(signatures, bandwidth=1.0):
    """Gaussian kernel on signature vectors."""
    n = len(signatures)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = signatures[i] - signatures[j]
            K[i, j] = np.exp(-np.dot(diff, diff) / (2 * bandwidth**2))
    return K

def rough_path_score(returns, macro_df, depth=4, bandwidth=1.0, alpha=0.1):
    """
    Compute predicted next-day return using kernel ridge regression on signature + macro.
    Train on past data (within the window), predict for the last point.
    """
    if len(returns) < 10 or macro_df is None or len(macro_df) < 10:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Use rolling window for local signatures
    window = 5
    if len(returns) < window + 1:
        return 0.0
    sigs = []
    targets = []
    macros_aligned = []
    for i in range(len(returns) - window):
        segment = returns[i:i+window]
        sig = signature(segment, depth)
        sigs.append(sig)
        targets.append(returns[i+window])
        macros_aligned.append(macro_df.iloc[i+window].values)
    if len(sigs) < 5:
        return 0.0
    sigs = np.array(sigs)
    targets = np.array(targets)
    macros = np.array(macros_aligned)
    # Remove any rows with NaN
    mask = ~(np.isnan(sigs).any(axis=1) | np.isnan(macros).any(axis=1) | np.isnan(targets))
    sigs = sigs[mask]
    macros = macros[mask]
    targets = targets[mask]
    if len(sigs) < 5:
        return 0.0
    # Standardise macro
    scaler = StandardScaler()
    macros_scaled = scaler.fit_transform(macros)
    # Compute kernel matrices
    K_sig = kernel_matrix(sigs, bandwidth)
    K_macro = kernel_matrix(macros_scaled, bandwidth)
    K = K_sig * K_macro
    # Ensure no NaN in K
    K = np.nan_to_num(K, nan=0.0)
    # Train kernel ridge
    kr = KernelRidge(alpha=alpha, kernel='precomputed')
    kr.fit(K, targets)
    # Predict for the last window
    last_segment = returns[-window:]
    last_sig = signature(last_segment, depth).reshape(1, -1)
    last_macro = macro_df.iloc[-1].values.reshape(1, -1)
    last_macro_scaled = scaler.transform(last_macro)
    # Compute kernel vector between last and all training points
    k_last = np.zeros(len(sigs))
    for i in range(len(sigs)):
        diff_sig = last_sig[0] - sigs[i]
        diff_macro = last_macro_scaled[0] - macros_scaled[i]
        k_last[i] = np.exp(-np.dot(diff_sig, diff_sig) / (2*bandwidth**2)) * np.exp(-np.dot(diff_macro, diff_macro) / (2*bandwidth**2))
    k_last = np.nan_to_num(k_last, nan=0.0)
    pred = kr.predict(k_last.reshape(1, -1))[0]
    return float(pred)
