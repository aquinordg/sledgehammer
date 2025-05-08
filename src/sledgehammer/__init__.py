"""
Python package `sledgehammer`: semantic evaluation for clustering results.

The package performs an evaluation of clustering results through
the semantic relationship between the significant frequent patterns
identified among the cluster items.

The method uses an internal validation technique to evaluate
the cluster rather than using distance-related metrics.
However, the algorithm requires that the data be organized in CATEGORICAL FORM.

Questions and information contact us: <aquinordga@gmail.com>
"""

import pandas as pd
import numpy as np
import math
import statistics
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def particularize_descriptors(descriptors, particular_threshold=1.0):
    """
    Particularization of descriptors based on support.

    This function particularizes descriptors using a threshold applied
    on the carrier (support maximum - support minimum) of the feature in the
    clusters.

    Parameters
    ----------
    particular_threshold: float
        Particularization threshold.  Given the relative support,
        0.0 means that the entire range of relative support will be used,
        while 0.5 will be used half, and 1.0 only maximum support is kept.

    descriptors: array-like of shape (n_clusters, n_features)
        Matrix with the support of features in each cluster.

    Returns
    -------
    descriptors: array-like of shape (n_clusters, n_features)
        Matrix with the computed particularized support of features in each
        cluster.
    """

    for feature in descriptors.columns:
        column = np.array(descriptors[feature])

        minimum_support = np.min(column)
        maximum_support = np.max(column)

        toremove = column < minimum_support + \
            particular_threshold * (maximum_support - minimum_support)
        descriptors.loc[toremove, feature] = 0.0

    return descriptors


def semantic_descriptors(X, labels, particular_threshold=None, report_form=False):
    """
    Semantic descriptors based on feature support.
    This function computes the support of the present feature (1-itemsets
    composed by the features with value 1) of the samples in each cluster.
    Features in a cluster that do not meet the *particularization criterion*
    have their support zeroed.
    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        Feature array of each sample.  All features must be binary.
    labels: array-like of shape (n_samples,)
        Cluster labels for each sample starting in 0.
    particular_threshold: {None, float}
        Particularization threshold.  `None` means no particularization
        strategy.
    Returns
    -------
    descriptors: array-like of shape (n_clusters, n_features)
        Matrix with the computed particularized support of features in each
        cluster.
    """

    n_clusters = max(labels) + 1

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])

    features = X.columns

    # 1-itemsets, for greater k we need a different algorithm
    support = X.groupby(labels).mean()

    if particular_threshold is not None:
        support = particularize_descriptors(
            support, particular_threshold=particular_threshold)
        
    if report_form is True:
        report = dict()
        for i in range(n_clusters):    
            report[i] = support.loc[i][support.loc[i] > 0].sort_values(ascending=False)
        return report
    else:
        return support


def sledgehammer_score_clusters(
        X,
        labels,
        W=[.3, .1, .5, .1],
        particular_threshold=None,
        aggregation='median'):
    """
    Computes the SLEDgeHammer (SLEDgeH) score for clusters in a dataset.

    The SLEDgeH score evaluates clusters based on four indicators: 
    Support (S), Length deviation (L), Exclusivity (E), and Descriptor support 
    Difference (D). These metrics are aggregated to provide an overall score for 
    each cluster, guiding the assessment of cluster quality.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Binary feature matrix where each row represents a sample and each column a feature.
    labels : array-like of shape (n_samples,)
        Cluster labels assigned to each sample. Labels should be integers starting from 0.
    W : array-like of shape (4,), optional (default=[0.3, 0.1, 0.5, 0.1])
        Weighting factors for the SLED indicators: Support (S), Length deviation (L),
        Exclusivity (E), and Difference (D), respectively.
    particular_threshold : float or None, optional (default=None)
        Threshold for particularization of descriptors. If `None`, no threshold is applied.
    aggregation : {'harmonic', 'geometric', 'median', None}, optional (default='median')
        Aggregation method for the SLED indicators:
        - 'harmonic': Harmonic mean.
        - 'geometric': Geometric mean.
        - 'median': Median value.
        - None: Returns the S, L, E, and D scores without aggregation.

    Returns
    -------
    scores : array-like of shape (n_clusters,)
        Aggregated SLEDgeH score for each cluster if `aggregation` is specified.
    score_matrix : array-like of shape (n_clusters, 4)
        Scores for S, L, E, and D for each cluster if `aggregation` is `None`.

    Example
    -------
    >>> from sklearn.cluster import KMeans
    >>> import numpy as np
    >>> X = np.random.randint(0, 2, (100, 5))  # Binary dataset
    >>> labels = KMeans(n_clusters=3, random_state=42).fit_predict(X)
    >>> scores, score_matrix = sledgehammer_score_clusters(X, labels, aggregation=None)
    >>> print(scores)
    >>> print(score_matrix)
    """


    n_clusters = max(labels) + 1
    descriptors = semantic_descriptors(
        X, labels, particular_threshold=particular_threshold).transpose()

    # S: Average support for descriptors (features with particularized support
    # greater than zero)
    def mean_gt_zero(x): return 0 if np.count_nonzero(
        x) == 0 else np.mean(x[x > 0])
    support_score = [mean_gt_zero(descriptors[cluster])
                     for cluster in range(n_clusters)]

    # L: Description set size deviation
    descriptor_set_size = np.array([np.count_nonzero(descriptors[cluster]) for
                                   cluster in range(n_clusters)])

    average_set_size = np.mean(descriptor_set_size[descriptor_set_size > 0])
    length_score = [0 if set_size == 0 else 1.0 /
                    (1.0 +
                     abs(set_size -
                         average_set_size)) for set_size in descriptor_set_size]

    # E: Exclusivity
    descriptor_sets = np.array([frozenset(
        descriptors.index[descriptors[cluster] > 0]) for cluster in range(n_clusters)])
    exclusive_sets = [
        descriptor_sets[cluster].difference(
            frozenset.union(
                *
                np.delete(
                    descriptor_sets,
                    cluster))) for cluster in range(n_clusters)]
    exclusive_score = [0 if len(descriptor_sets[cluster]) == 0 else len(
        exclusive_sets[cluster]) / len(descriptor_sets[cluster]) for cluster in range(n_clusters)]

    # D: Maximum ordered support difference
    ordered_support = [np.sort(descriptors[cluster])
                       for cluster in range(n_clusters)]
    diff_score = [math.sqrt(np.max(np.diff(ordered_support[cluster])))
                  for cluster in range(n_clusters)]
        
    score = pd.DataFrame.from_dict({'S': [W[0] * s for s in support_score],
                                    'L': [W[1] * l for l in length_score],
                                    'E': [W[2] * e for e in exclusive_score],
                                    'D': [W[3] * d for d in diff_score]})

    if aggregation == 'harmonic':
        score = score.transpose().apply(statistics.harmonic_mean)
    elif aggregation == 'geometric':
        score = score.transpose().apply(statistics.geometric_mean)
    elif aggregation == 'median':
        score = score.transpose().apply(statistics.median)
    else:
        assert aggregation is None

    return score


def sledgehammer_score(
        X,
        labels,
        W=[.3, .1, .5, .1],
        particular_threshold=None,
        aggregation='median'):
    """
    Computes the average SLEDgeH score for all clusters.

    This function calculates the SLEDgeH score, which evaluates the quality of clusters 
    based on four indicators: Support (S), Length deviation (L), Exclusivity (E), 
    and Descriptor support Difference (D). The scores for each cluster are computed 
    and then averaged to provide an overall measure of clustering quality.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Binary feature matrix where each row represents a sample and each column a feature.
    labels : array-like of shape (n_samples,)
        Cluster labels assigned to each sample. Labels should be integers starting from 0.
    W : array-like of shape (4,), optional (default=[0.3, 0.1, 0.5, 0.1])
        Weighting factors for the SLED indicators: Support (S), Length deviation (L),
        Exclusivity (E), and Difference (D), respectively.
    particular_threshold : float or None, optional (default=None)
        Threshold for particularization of descriptors. If `None`, no threshold is applied.
    aggregation : {'harmonic', 'geometric', 'median'}, optional (default='median')
        Aggregation method for the SLED indicators in each cluster:
        - 'harmonic': Harmonic mean of S, L, E, and D indicators.
        - 'geometric': Geometric mean of S, L, E, and D indicators.
        - 'median': Median of S, L, E, and D indicators.

    Returns
    -------
    score : float
        The average SLEDgeH score across all clusters.

    Example
    -------
    >>> from sklearn.cluster import KMeans
    >>> import numpy as np
    >>> from sledgehammer import sledgehammer_score
    >>> X = np.random.randint(0, 2, (100, 5))  # Binary dataset
    >>> labels = KMeans(n_clusters=3, random_state=42).fit_predict(X)
    >>> average_score = sledgehammer_score(X, labels, aggregation='harmonic')
    >>> print(average_score)
    """

    assert aggregation is not None
    return np.mean(
        sledgehammer_score_clusters(
            X,
            labels,
            W,
            particular_threshold=particular_threshold,
            aggregation=aggregation))
            
def sledge_curve(X, labels, particular_threshold=0.0, aggregation='harmonic'):
    """
    SLEDge curve.

    This function computes the SLEDge curve.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        Feature array of each sample.  All features must be binary.
    labels: array-like of shape (n_samples,)
        Cluster labels for each sample starting in 0.
    particular_threshold: {None, float}
        Particularization threshold.  `None` means no particularization
        strategy.
    aggregation: {'harmonic', 'geometric', 'median', None}
        Strategy to aggregate values of *S*, *L*, *E*, and *D*.

    Returns
    -------
    fractions: array-like of shape (>2,)
        Decreasing rate that element `i` is the fraction of clusters with
        SLEDge score >= `thresholds[i]`.  `fractions[0]` is always `1`.
    thresholds: array-like of shape (>2, )
        Increasing thresholds of the cluster SLEDge score used to compute
        `fractions`.  `thresholds[0]` is always `0` and `thresholds[-1]` is
        always `1`.
    """
    scores = sledge_score_clusters(X, labels,
                                   particular_threshold=particular_threshold,
                                   aggregation=aggregation)
    n_clusters = len(scores)

    thresholds = np.unique(scores)
    if thresholds[0] != 1:
        thresholds = np.concatenate((thresholds, [1]))
    if thresholds[len(thresholds) - 1] != 0:
        thresholds = np.concatenate(([0], thresholds))

    fractions = np.array(
        [np.count_nonzero(scores >= thr) / n_clusters for thr in thresholds])

    return fractions, thresholds


##### FORGE Clustering #####

from heapq import nsmallest

def get_sledgehammer_score(target, ref):
    """
    Compute the sledgehammer dissimilarity score between target and reference.

    This score measures how distinct a target sample is from a reference group,
    with higher values indicating greater dissimilarity.

    Parameters
    ----------
    target : array-like of shape (n_features,)
        Binary feature vector of the target sample.
    ref : array-like of shape (n_ref_samples, n_features)
        Binary feature matrix of reference samples.

    Returns
    -------
    score : float
        Sledgehammer dissimilarity score between 0 and 1, where:
        - 0: target is identical to reference
        - 1: target is maximally distinct from reference

    Notes
    -----
    1. Both target and reference must contain only binary values (0 or 1).
    2. The score combines four components (S, L, E, D) that measure different
       aspects of feature relationships.
    3. For empty reference, returns 1 (maximal dissimilarity).

    Examples
    --------
    >>> target = np.array([1, 0, 1])
    >>> ref = np.array([[0, 1, 1], [1, 1, 0]])
    >>> score = get_sledgehammer_score(target, ref)
    >>> print(f"{score:.3f}")
    0.742
    """
    # Garante que 'target' e 'ref' sejam 2D (mesmo se tiverem apenas 1 linha)
    target = np.atleast_2d(target)
    ref = np.atleast_2d(ref)

    # Concatena ref e target em X (target fica na última posição)
    X = np.vstack([ref, target])
    
    # Cria labels: 0 para ref, 1 apenas para a última linha (target)
    labels = np.zeros(len(X), dtype=int)
    labels[-1] = 1  # Último elemento = 1 (target)
    
    return sledgehammer_score(X, labels, aggregation='median')

def forge(data, k):
    """
    FORGE (Feature-Oriented Robust Grouping Engine).

    This function performs binary data clustering using a sledgehammer-score-based approach.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Binary feature array where each row represents a sample and each column a feature.
        All features must be binary (0 or 1).
    k : int
        Number of clusters to form (must be >= 1).

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels for each sample, starting from 0 to k-1.

    Notes
    -----
    1. The algorithm works best with high-dimensional binary data.
    2. For k=1, all samples are assigned to cluster 0.
    3. The distance metric used is the sledgehammer score.

    Examples
    --------
    >>> from forge import FORGE
    >>> data = np.array([[1,0,1], [0,1,1], [1,1,0]])
    >>> labels = FORGE(data, k=2)
    >>> print(labels)
    [0 1 0]
    """
    
    n_samples = len(data)
    if k == 1:
        return np.zeros(n_samples, dtype=int)
    
    # 1. Calcula a matriz de dissimilaridade completa
    dissim_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            dissim_matrix[i,j] = dissim_matrix[j,i] = get_sledgehammer_score(data[i], data[j])
    
    # 2. Seleciona os k exemplares mais distintos
    exemplars = []
    remaining_points = set(range(n_samples))
    
    # Primeiro exemplar: ponto com maior dissimilaridade média
    avg_dissim = dissim_matrix.mean(axis=1)
    exemplars.append(np.argmax(avg_dissim))
    remaining_points.remove(exemplars[0])
    
    # Demais exemplares: pontos mais distantes dos já selecionados
    for _ in range(1, k):
        max_min_dist = -1
        best_point = -1
        for point in remaining_points:
            min_dist = min(dissim_matrix[point, e] for e in exemplars)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_point = point
        exemplars.append(best_point)
        remaining_points.remove(best_point)
    
    # 3. Atribui cada ponto ao exemplar mais próximo
    labels = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        closest_exemplar = exemplars[0]
        min_dist = dissim_matrix[i, closest_exemplar]
        for j in range(1, k):
            if dissim_matrix[i, exemplars[j]] < min_dist:
                min_dist = dissim_matrix[i, exemplars[j]]
                closest_exemplar = exemplars[j]
        labels[i] = exemplars.index(closest_exemplar)
    
    # 4. Renumera os clusters para 0 a k-1
    unique_labels = np.unique(labels)
    label_mapping = {old: new for new, old in enumerate(unique_labels)}
    return np.array([label_mapping[l] for l in labels])