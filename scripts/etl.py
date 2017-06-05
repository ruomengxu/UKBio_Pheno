import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from util import get_output_dir

def load_sparse_feature_matrix():
    medication_df = pd.read_csv('{}/MEDICATIONS/part-00000'.format(get_output_dir()), dtype={'featureName': str})
    cancer_df = pd.read_csv('{}/CANCERS/part-00000'.format(get_output_dir()), dtype={'featureName': str})
    non_cancer_illness_df = pd.read_csv('{}/NON_CANCER_ILLNESSES/part-00000'.format(get_output_dir()), dtype={'featureName': str})
    lifestyle_df = pd.read_csv('{}/LIFESTYLE/part-00000'.format(get_output_dir()), dtype={'featureName': str})
    demographic_df = pd.read_csv('{}/DEMOGRAPHIC/part-00000'.format(get_output_dir()), dtype={'featureName': str})
    physical_measure_df = pd.read_csv('{}/PHYSICALMEASURE/part-00000'.format(get_output_dir()), dtype={'featureName': str})

    df = non_cancer_illness_df \
        .append(cancer_df) \
        .append(medication_df) \
        # # .append(physical_measure_df)\
        # # .append(lifestyle_df)\
        # # .append(demographic_df)\

    cols = ['featureName', 'featureValue']
    df = df.sort_values(by='eid')
    grouped = df.groupby('eid')
    l = [dict(zip(*g[cols].T.values)) for n, g in grouped]

    vec = DictVectorizer()
    sparse_feature_matrix = vec.fit_transform(l)

    eids = [group['eid'].unique()[0] for _, group in grouped]

    feature_names = vec.get_feature_names()

    return eids, sparse_feature_matrix, feature_names
