import numpy as np
from scipy.stats import norm as gaussian

def sample_along_axis(mat, axis=-1):
    totals = np.cumsum(mat, axis=axis)
    idx = np.random.random(totals.shape[:-1])[...,None]
    return np.argmax(totals >= idx, axis=axis)

def sharpen_to_onehot(values, n_idx, axis=-1, round=8):
    if round: values = np.around(values, round)
    idx = np.argmax(values, axis)
    mat = np.eye(n_idx)[idx]
    if axis != -1: mat = np.rollaxis(mat, -1, axis)
    return mat

def gaussian_cdf(x):
    dist = gaussian()
    return dist.cdf(x)

def exponential_interpolate(n, max):
    return sorted({int(x) for x in np.exp(np.log(max) * (np.array(range(n))+1)/n)})