""" functions for processing information theoretic diagnostics """

import numpy as np
from collections import Counter
from tqdm import tqdm

def discretize(vals, bins=20):
    """ bin continuous values """
    binned = np.digitize(vals, np.linspace(np.min(vals), np.max(vals), bins))
    return binned


def entropy(vals, bins=20):
    """ calculates entropy of a series """

    disc_vals = discretize(vals, bins)
    counts = Counter(disc_vals)
    n = len(vals)
    probs = np.array(list(counts.values())) / n
    return -np.sum(probs * np.log2(probs))

def joint_entropy(variables, bins):
    """ calculates joint entropy between prices, volume data """

    disc_var = [discretize(v, bins) for v in variables]

    tuples = list(zip(*disc_var))

    counts = Counter(tuples)
    n = len(tuples)

    probs = np.array(list(counts.values())) / n

    return -np.sum(probs * np.log2(probs))

def conditional_mutual_info(volume, prices, signals, bins=20):
    """ calc. conditional mutual info between prices, volume given signals """

    joint_entropy_vs = joint_entropy([volume, signals], bins)
    # print("H(V, S): " + str(joint_entropy_vs))
    joint_entropy_ps = joint_entropy([prices, signals], bins)
    # print("H(P, S): " + str(joint_entropy_ps))
    s_entropy = entropy(signals, bins)
    # print("H(S): " + str(s_entropy))
    joint_entropy_vsp = joint_entropy([volume, signals, prices], bins)
    # print("H(V,S,P): " + str(joint_entropy_vsp))

    return joint_entropy_vs + joint_entropy_ps - s_entropy - joint_entropy_vsp

def mutual_information(x, y, bins=20):
    """ Calculate mutual information between two signals """
    h_x = entropy(x, bins)
    h_y = entropy(y, bins)
    h_xy = joint_entropy([x, y], bins)
    return h_x + h_y - h_xy

def rolling_mutual_entropy(x, y, window_size, step=1, bins=20):
    n = len(x)
    mi_values = []
    centers = []
    
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        x_window = x[start:end]
        y_window = y[start:end]
        
        mi = mutual_information(x_window, y_window, bins)
        mi_values.append(mi)
        centers.append(start + window_size // 2)
    
    return np.array(mi_values), np.array(centers)

def rolling_entropy(x, window_size, step, bins=20):
    n = len(x)
    n_windows = (n - window_size) // step + 1
    entropy_values = np.zeros(n_windows)
    
    for w in range(n_windows):
        start = w * step
        end = start + window_size
        entropy_values[w] = entropy(x[start:end], bins)
    
    return entropy_values

def rolling_conditional_mutual_information(x, y, z, window_size, step, bins=20):
    n = len(x)
    cmi_values = []
    centers = []

    
    pbar = tqdm(range(0, n - window_size + 1, step), desc="Window count...")

    for start in pbar:
        end = start + window_size
        
        x_window = x[start:end]
        y_window = y[start:end]
        z_window = z[start:end]
        
        cmi = conditional_mutual_info(x_window, y_window, z_window, bins)
        cmi_values.append(cmi)
        centers.append(start + window_size // 2)
    
    return np.array(cmi_values), np.array(centers)