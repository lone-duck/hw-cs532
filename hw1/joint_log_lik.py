import numpy as np
from scipy.special import loggamma as lgamma

def joint_log_lik(doc_counts, topic_counts, alpha, gamma):
    """
    Calculate the joint log likelihood of the model
    
    Args:
        doc_counts: n_docs x n_topics array of counts per document of unique topics
        topic_counts: n_topics x alphabet_size array of counts per topic of unique words
        alpha: prior dirichlet parameter on document specific distributions over topics
        gamma: prior dirichlet parameter on topic specific distribuitons over words.
    Returns:
        ll: the joint log likelihood of the model
    """

    D = doc_counts.shape[0]
    K, W = topic_counts.shape


    # compute contribution from uniform vectors first
    log_B_alpha = K*lgamma(alpha) - lgamma(K*alpha)
    log_B_beta = W*lgamma(gamma) - lgamma(W*gamma)

    nd_plus_alpha = doc_counts + alpha
    nk_plus_gamma = topic_counts + gamma

    log_B_nd_plus_alpha = log_beta(nd_plus_alpha)
    log_B_nk_plus_gamma = log_beta(nk_plus_gamma)

    ll = np.sum(log_B_nd_plus_alpha) + np.sum(log_B_nk_plus_gamma) - D*log_B_alpha - K*log_B_beta

    return ll


def log_beta(A):
    """
    Calculate multinomial beta function on rows of a matrix.

    Args:
        A: input matrix, shape N x M
    Returns:
        logBeta: multinomial beta function computed on each row, shape (N,)
    """
    logBeta = np.sum(lgamma(A), axis=1) - lgamma(np.sum(A, axis=1))
    return logBeta
