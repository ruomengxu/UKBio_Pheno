from __future__ import print_function
import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
import pandas as pd

import util

def get_nmf_w_h(feature_matrix, n_components=4, alpha=0.0, l1_ratio=0.0, max_iter=200):
    model = NMF(n_components=n_components, init='nndsvd', alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)
    W = model.fit_transform(feature_matrix)
    H = model.components_
    reconstruction_err = model.reconstruction_err_
    return W, H, reconstruction_err

def get_cluster_assignments(eids, W):
    return list(zip(eids, np.argmax(W, axis=1)))

def get_optimal_k(eids, feature_matrix, distance_metric='euclidean'):
    silhouette_scores = []
    for k in range(2, 21, 4):
        scores = []
        for iter in range(10):
            W, _, _ = get_nmf_w_h(feature_matrix, n_components=k)
            cluster_assignments = get_cluster_assignments(eids, W)
            silhouette_score = get_silhouette_score(feature_matrix, cluster_assignments, distance_metric)
            scores.append(silhouette_score)
        silhouette_scores.append((sum(scores) / len(scores), k))
        print(k, sum(scores) / len(scores))
    return sorted(silhouette_scores, reverse=True)[0][1]

def get_silhouette_score(feature_matrix, cluster_assignments, distance_metric='euclidean'):
    labels = [assignment for eid, assignment in cluster_assignments]
    return metrics.silhouette_score(feature_matrix, labels, distance_metric, sample_size=30000)

def get_count_features_above_threshold(H, threshold):
    return (H > threshold).sum(1)

def get_nonzero_featurenames(H, threshold, feature_names):
    r, c = np.where(H > threshold)
    indices = np.split(c, np.flatnonzero(r[1:] > r[:-1])+1)
    cluster_feature_name_map = {}
    for i, name in enumerate(feature_names):
        for ii, row in enumerate(indices):
            if np.any(row == i):
                cluster_feature_name_map['label'] = cluster_feature_name_map.get('label', []) + [ii]
                cluster_feature_name_map['featureName'] = cluster_feature_name_map.get('featureName', []) + [name]
                cluster_feature_name_map['weight'] = cluster_feature_name_map.get('weight', []) + [H[ii, i]]
    return cluster_feature_name_map

def hyperparameter_tuning(feature_matrix):
    alphas = [1, 5, 15]
    l1_ratios = [0, .5, 1]
    ks = range(2, 40, 5)
    threshold = 1

    d = {}
    for k in ks:
        for alpha in alphas:
            for l1_ratio in l1_ratios:
                    W, H, _ = get_nmf_w_h(feature_matrix, n_components=k, alpha=alpha, l1_ratio=l1_ratio)
                    avg_count_above_threshold = np.mean(get_count_features_above_threshold(H, threshold))
                    num_labels = len(set([label for _, label in get_cluster_assignments(np.zeros(W.shape[0]), W)]))
                    d['k'] = d.get('k', []) + [k]
                    d['alpha'] = d.get('alpha', []) + [alpha]
                    d['l1_ratio'] = d.get('l1_ratio', []) + [l1_ratio]
                    d['num_labels'] = d.get('num_labels', []) + [num_labels]
                    d['avg_nonzero_feature_count'] = d.get('avg_nonzero_feature_count', []) + [avg_count_above_threshold]
                    print('k={0}, alpha={1}, l1_ratio={2}, num_labels={3}, count={4}'.format(k, alpha, l1_ratio, num_labels, avg_count_above_threshold))
    #output all results
    df = pd.DataFrame(d)
    df.to_csv('{}/hyperparamtuning.csv'.format(util.get_output_dir()))

    #output optimal results
    df = df[(df['count'] >= 5) & (df['count'] <= 10)]
    df.to_csv('{}/optimalparams.csv'.format(util.get_output_dir()))

def feature_selection(X, p):
    sel = VarianceThreshold(threshold=p * (1 - p))
    print('before feature selection: {} features'.format(X.shape[1]))
    X_after_feature_selection =  sel.fit_transform(X)
    print('after feature selection: {} features'.format(X_after_feature_selection.shape[1]))
    return X_after_feature_selection,sel.get_support(indices=True)