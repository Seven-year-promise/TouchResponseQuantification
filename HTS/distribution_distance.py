import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance
from scipy import stats
import math


def ave_distance(p, q):
    distance = []
    for p_item in p:
        for q_item in q:
            distance.append(abs(p_item-q_item))
    if np.mean(p) > np.mean(q):
        sign = -1
    else:
        sign = 1
    return np.mean(distance)*sign

_SQRT2 = np.sqrt(2)     # sqrt(2) with default precision np.float64

def hellinger1(p, q):
    return norm(np.sqrt(p) - np.sqrt(q)) / _SQRT2

def hellinger2(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2

def hellinger3(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2

def wasserstein(p_pdf, q_pdf):
    return wasserstein_distance(p_pdf, q_pdf)

def wasserstein_2_1d(p, q):
    """
    :param data_a:
    :param data_b:
    :return ||m_1 - m_2||_2^2 + trace(c_1+c_2-2*(c_2^(0.5)*c_1*c_2^(0.5))^(0.5))
    """
    sign = 1
    mu_p = np.mean(p)
    mu_q = np.mean(q)
    if mu_p > mu_q:
        sign = -1

    c_p = np.cov(p)
    c_q = np.cov(q)

    return sign*((mu_p-mu_q)**2 + c_p + c_q -2*np.sqrt(np.sqrt(c_q)*c_p*np.sqrt(c_q)))

def compute_prob_distance(p, q, algorithm=""):
    p_pdf = stats.norm.ppf(p)
    q_pdf = stats.norm.ppf(q)

    if algorithm=="ave":
        return ave_distance(p, q)
    elif algorithm=="hellinger":
        return hellinger3(p_pdf, q_pdf)
    elif algorithm=="wasserstein":
        return wasserstein(p_pdf, q_pdf)
    elif algorithm=="wasserstein_2":
        return wasserstein_2_1d(p, q)
    else:
        raise NotImplemented