import clustering

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from sklearn.decomposition import TruncatedSVD
from sklearn import metrics

import numpy as np

import pandas as pd

import util

import random

def pca_projected_clusters(feature_matrix):
    n_components = 5

    if not util.is_server():
        color_names = colors.cnames

        df = pd.read_csv('{}/pca_projected_clusters.csv'.format(util.get_output_dir()))
        df = df[['x', 'y', 'label']]
        ax = df[df['label'] == 0].plot(kind='scatter', x='x', y='y', c=random.choice(list(color_names.keys())),
                                       label='Phenotype 0')
        for k in range(1, n_components):
            cluster = df[df['label'] == k]
            if cluster.shape[0] > 0:
                cluster.plot(kind='scatter', x='x', y='y', c=random.choice(list(color_names.keys())),
                             label='Phenotype {}'.format(k), ax=ax)
        plt.title('PCA Projected Clusters')
        plt.savefig('{}/pca_projected_clusters.png'.format(util.get_output_dir()))
        plt.clf()
        return

    W, H, _ = clustering.get_nmf_w_h(feature_matrix, n_components=n_components, alpha=5, l1_ratio=1.0)
    labels = [label for eid, label in clustering.get_cluster_assignments(np.zeros(W.shape[0]), W)]
    model = TruncatedSVD(n_components=2)
    data_r = model.fit_transform(feature_matrix)
    d = {}
    d['x'] = [data[0] for data in data_r]
    d['y'] = [data[1] for data in data_r]
    d['label'] = labels
    df = pd.DataFrame(d)
    df.to_csv('{}/pca_projected_clusters.csv'.format(util.get_output_dir()))



def k_vs_silhouette(feature_matrix, distance_metric='euclidean'):

    if not util.is_server():
        df = pd.read_csv('{}/silhouette_avg.csv'.format(util.get_output_dir()))
        df = df[['k', 'silhouette_avg']].set_index('k')
        df.plot(title='K vs Silhouette Average')
        plt.xlabel('K')
        plt.ylabel('Silhouette average')
        plt.show()
        return

    silhouette_avgs = []
    ks = range(2, 40, 4)
    for k in ks:
        W, H, _ = clustering.get_nmf_w_h(feature_matrix, n_components=k)
        labels = [label for eid, label in clustering.get_cluster_assignments(np.zeros(W.shape[0]), W)]
        for iter in range(3):
            avgs = []
            avgs.append(metrics.silhouette_score(feature_matrix, labels, distance_metric,
                                                      sample_size=10000))
        silhouette_avgs.append(sum(avgs) / len(avgs))
    d = {}
    d['k'] = ks
    d['silhouette_avg'] = silhouette_avgs
    df = pd.DataFrame(d)
    df.to_csv('{}/silhouette_avg.csv'.format(util.get_output_dir()))

def silhouette_plot(eids, feature_matrix, distance_metric='euclidean'):
    for k in range(2, 21, 4):
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, feature_matrix.shape[0] + (k + 1) * 10])

        W, H, _ = clustering.get_nmf_w_h(feature_matrix, n_components=k)
        cluster_assignments = clustering.get_cluster_assignments(eids, W)
        cluster_labels = np.array([assignment for eid, assignment in cluster_assignments])

        sample_size = 30000
        indices = np.random.permutation(feature_matrix.shape[0])[:sample_size]
        feature_matrix_sample = feature_matrix[indices]
        cluster_labels_sample = cluster_labels[indices]
        sample_silhouette_values = metrics.silhouette_samples(feature_matrix_sample, cluster_labels_sample, distance_metric)

        silhouette_avg = metrics.silhouette_score(feature_matrix, cluster_labels, distance_metric, sample_size=sample_size)

        #project data onto 2 dimensions for visualization
        model = TruncatedSVD(n_components=2)
        data_r = model.fit_transform(feature_matrix)

        y_lower = 10
        for i in range(k):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / k)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.spectral(cluster_labels.astype(float) / k)
        ax2.scatter(data_r[:, 0], data_r[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors)

        # # Labeling the clusters
        # centers = H
        # # Draw white circles at cluster centers
        # ax2.scatter(centers[:, 0], centers[:, 1],
        #             marker='o', c="white", alpha=1, s=200)
        #
        # for i, c in enumerate(centers):
        #     ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for NMF clustering"
                      "with n_clusters = %d" % k),
                     fontsize=14, fontweight='bold')

        plt.savefig('{0}/silAnalysis_k={1}.png'.format(util.get_output_dir(), k))
        plt.clf()

def iteration_vs_error(feature_matrix, n_components):
    iterations = range(10, 300, 10)
    errors = []
    for iteration in iterations:
        _, _, error = clustering.get_nmf_w_h(feature_matrix, n_components, max_iter=iteration)
        errors.append(error)

    d = {'error': errors, 'iteration': iterations}
    df = pd.DataFrame(d)
    df.plot(x='iteration', y='error')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.savefig('{}/iteration_vs_error.png'.format(util.get_output_dir()))
    plt.clf()

def percentage_feature_in_cluster_vs_cohort(mode, labels, H, feature_names, pop_size):
    valid_modes = ['medication', 'non-cancer-illness', 'cancer']
    if mode not in valid_modes:
        raise AttributeError('mode must be one of {}'.format(', '.join(valid_modes)))

    if not util.is_server():
        df = pd.read_csv('{0}/{1}_percent_ftr_vs_clstr.csv'.format(util.get_output_dir(), mode))
        pre = 'percentFeatureInCluster'
        d1 = df.set_index(['meaning', 'label'])[pre].unstack(fill_value=0).add_prefix(pre).reset_index()
        df = d1.set_index('meaning').join(df[['meaning', 'percentFeatureInPop']].set_index('meaning'), how='inner').drop_duplicates()
        df.plot.bar(title='Percent Feature in Cluster Vs Pop', fontsize=8, rot=90, grid=True, stacked=True, figsize=[20, 20], colormap=plt.cm.ocean)
        plt.xlabel(mode)
        plt.gcf().subplots_adjust(bottom=0.5)
        plt.savefig('{0}/{1}_percent_ftr_vs_clstr.png'.format(util.get_output_dir(), mode), bbox_inches='tight')
        plt.clf()
        return

    medication_df = pd.read_csv('{}/MEDICATIONS/part-00000'.format(util.get_output_dir()), dtype={'featureName': str})
    cancer_df = pd.read_csv('{}/CANCERS/part-00000'.format(util.get_output_dir()), dtype={'featureName': str})
    non_cancer_illness_df = pd.read_csv('{}/NON_CANCER_ILLNESSES/part-00000'.format(util.get_output_dir()), dtype={'featureName': str})

    if mode == 'medication':
        df = medication_df
        info_df = pd.read_csv('{}/medicationCodes/coding4.tsv'.format(util.get_output_dir()),
                              delimiter='\t',
                              dtype={'coding': str},
                              usecols=['coding', 'meaning'])
        join_on = 'coding'
    elif mode == 'non-cancer-illness':
        df = non_cancer_illness_df
        info_df = pd.read_csv('{}/nonCancerIllnessCodes/coding6.tsv'.format(util.get_output_dir()),
                              delimiter='\t',
                              dtype={'node_id': str},
                              usecols=['node_id', 'meaning'])
        join_on = 'node_id'
    elif mode == 'cancer':
        df = cancer_df
        info_df = pd.read_csv('{}/cancerCodes/coding3.tsv'.format(util.get_output_dir()),
                              delimiter='\t',
                              dtype={'node_id': str},
                              usecols=['node_id', 'meaning'])
        join_on = 'node_id'

    nonzero_feature_names = clustering.get_nonzero_featurenames(H, 1, feature_names)
    feature_names_df = pd.DataFrame(nonzero_feature_names).drop('weight', axis=1)


    #join data with cluster labels
    cluster_labels_df = pd.DataFrame.from_records(labels, columns=['eid', 'label']).set_index('eid')
    data = df.join(cluster_labels_df, on='eid', how='inner')

    num_patients_in_cluster_df = pd.DataFrame(data.groupby(by='label')['eid']
                                              .apply(lambda x: len(x.unique()))
                                              .rename('numPatientsInCluster'))

    count_feature_in_pop_df = pd.DataFrame(data.groupby(by='featureName')['eid']
                                           .apply(lambda x: len(x.unique()))
                                           .rename('countFeatureInPop'))

    features_selected_by_H_df = data\
        .set_index(['label', 'featureName'])\
        .join(feature_names_df.set_index(['label', 'featureName']), how='inner')\
        .reset_index()
    feature_count_in_cluster_df = pd.DataFrame(features_selected_by_H_df
                                               .groupby(by=['label', 'featureName'])['eid']
                                               .apply(lambda x: len(x.unique()))
                                               .rename('featureCountInCluster'))

    data = features_selected_by_H_df.set_index(['label', 'featureName']).join(feature_count_in_cluster_df, how='inner').reset_index()
    data = data.set_index('label').join(num_patients_in_cluster_df, how='inner').reset_index()
    data = data.set_index('featureName').join(count_feature_in_pop_df, how='inner').reset_index()

    #calculate count of patients with feauture in cluster / num patients in cluster
    data['percentFeatureInCluster'] = data['featureCountInCluster'] / data['numPatientsInCluster'] * 100
    #calculate total count of feature / total pop
    data['percentFeatureInPop'] = data['countFeatureInPop'] / pop_size * 100
    #get meaning names
    data['coding_or_node_id'] = data['featureName'].apply(lambda str: str.split('=')[1])
    data = data.set_index('coding_or_node_id').join(info_df.set_index(join_on), how='inner').reset_index()
    data = data[['meaning', 'label', 'percentFeatureInCluster', 'percentFeatureInPop']].drop_duplicates()
    data.to_csv('{0}/{1}_percent_ftr_vs_clstr.csv'.format(util.get_output_dir(), mode))



def vis_grs(labels):
    grs_df = pd.read_csv('{}/GWAS_Catalog_20170317_50traits_Good_CALCULATED_GRS.txt'.format(util.get_output_dir()),sep='\t')
    cluster_assignment_df = pd.DataFrame.from_records(labels)
    cluster_assignment_df.columns = ['Indiv', 'cluster']
    grs_cluster_df = grs_df.join(cluster_assignment_df.set_index('Indiv'), on = 'Indiv')
    grs_cluster_df = grs_cluster_df[pd.notnull(grs_cluster_df['cluster'])].drop('Indiv',axis = 1)
    grs_cluster_df_group = grs_cluster_df.groupby(["cluster"])
    matrix = grs_cluster_df_group.mean().as_matrix()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.ocean, aspect = 'auto')
    plt.colorbar()
    plt.ylabel('Phenotype')
    plt.xlabel('Disease')
    plt.title('Average Genetic Risk Score of Each Clutser ')
    plt.savefig('{}/grs.png'.format(util.get_output_dir()))
    plt.clf()


def vis_feature_bar(feature_matrix):
    frequency = feature_matrix.sum(axis = 0)
    bar_num = frequency.shape[1]
    plt.bar(range(0,bar_num),frequency.getA().flatten())
    plt.ylabel('Frequency')
    plt.xlabel('Feature#')
    plt.title('Frequency of Features')
    plt.savefig('{}/frequency.png'.format(util.get_output_dir()))
    plt.clf()

def vis_info_table(H, feature_names):
    nonzero_feature_names = clustering.get_nonzero_featurenames(H, .1, feature_names)
    feature_names_weight_df = pd.DataFrame(nonzero_feature_names).set_index('featureName')
    med_info_df = pd.read_csv('{}/medicationCodes/coding4.tsv'.format(util.get_output_dir()),
                          delimiter='\t',
                          dtype={'coding': str},
                          usecols=['coding', 'meaning']).rename(columns = {'coding':'node_id'})
    med_info_df['node_id'] = '20003='+ med_info_df['node_id'].astype(str)
    med_info_df['category'] = 'med'
    med_info = feature_names_weight_df\
     .join(med_info_df.set_index('node_id'), how='inner')\
     .reset_index()\
     .drop('index',axis = 1)

    noncan_info_df = pd.read_csv('{}/nonCancerIllnessCodes/coding6.tsv'.format(util.get_output_dir()),
                          delimiter='\t',
                          dtype={'node_id': str},
                          usecols=['node_id', 'meaning']) 
    noncan_info_df['node_id'] = '20002=' + noncan_info_df['node_id'].astype(str)
    noncan_info_df['category'] = 'noncan'
    noncan_info = feature_names_weight_df\
     .join(noncan_info_df.set_index('node_id'), how='inner')\
     .reset_index()\
     .drop('index',axis = 1)

    can_info_df = pd.read_csv('{}/cancerCodes/coding3.tsv'.format(util.get_output_dir()),
                          delimiter='\t',
                          dtype={'node_id': str},
                          usecols=['node_id', 'meaning'])
    can_info_df['node_id'] = '20001=' + can_info_df['node_id'].astype(str)
    can_info_df['category'] = 'cancer'
    can_info = feature_names_weight_df\
     .join(can_info_df.set_index('node_id'), how='inner')\
     .reset_index()\
     .drop('index',axis = 1)
    frame = [med_info, noncan_info, can_info]
    f = open('{}/info_table.csv'.format(util.get_output_dir()),'w+')
    info_table = pd.concat(frame).groupby(['label','category'])\
        .apply(lambda x: x.loc[:,['label','category','meaning','weight']]\
        .to_csv(f,index = False))

#figure 6 in paper
def nonzero_threshold_plot(H,feature_names,selected_indices):
    med_index = []
    cancer_index = []
    noncan_index = []
    feature_names = np.asarray(feature_names)[selected_indices]
    for index, item in enumerate(feature_names):
        if '20001' in item:
            cancer_index.append(index)
        elif '20002' in item:
            noncan_index.append(index)
        else:
            med_index.append(index)
    H_med = H[:,med_index]
    H_can = H[:,cancer_index]
    H_noncan = H[:, noncan_index]
    #need to be modified after training the model.
    threshold_list = [0,0.0001,0.01,0.05,0.1]
    threshold_list_c = [0,0.0005,0.001,0.005,0.05]
    med = np.array([clustering.get_count_features_above_threshold(H_med,threshold) for threshold in threshold_list])
    can = np.array([clustering.get_count_features_above_threshold(H_can,threshold) for threshold in threshold_list_c])
    noncan = np.array([clustering.get_count_features_above_threshold(H_noncan,threshold) for threshold in threshold_list])
    group_bar(med,threshold_list,range(0,5),'Medication')
    group_bar(can,threshold_list_c,range(0,5),'Cancer')
    group_bar(noncan,threshold_list,range(0,5),'NonCancer Illness')

def nonzero_alpha_beta_plot(k):
    hyperpara_df = pd.read_csv('{}/hyperparamtuning.csv'.format(util.get_output_dir()),
                              dtype={'k':int, 'alpha':float,'beta':int,'count':int},
                              usecols=['k', 'alpha','beta','count'])
    alphas = [.01, .1, 1, 10, 100]
    betas = [1, 10, 100, 1000]
    hyperpara_df = hyperpara_df[hyperpara_df['k']==k]
    alpha_beat_array =[]
    for alpha in alphas:
        temp = []
        for beta in betas:
            index = np.logical_and(hyperpara_df['alpha']==alpha,hyperpara_df['beta']==beta)
            temp.append(hyperpara_df[index]['count'].values[0])
        alpha_beat_array.append(temp)
    alpha_beat_array = np.asarray(alpha_beat_array)
    group_bar(alpha_beat_array,alphas,betas,'Alpha-Beta-Count Bar Plot')

# Need to adjust width and figsize if parameters are changed
def group_bar(data,x_ticklabel,legend,category):
    width = 0.15
    fig, ax = plt.subplots(figsize=(15,10))
    pos = range(0,data.shape[0])
    times = range(0,data.shape[1])
    for i in times:
        postion = [ p + i*width for p in pos]
        plt.bar(postion,data[:,i],width,alpha=0.5)
    ax.set_ylabel("Count")
    ax.set_xticks([p + width*(data.shape[1]-1)/2 for p in pos ])
    ax.set_xticklabels(x_ticklabel)
    ax.set_title(category)
    plt.legend(["cluster"+str(i) for i in pos],loc = "upper left")
    plt.ylim([0, max(data[0,:])] )
    if category in ['Medication','Cancer','NonCancer Illness']:
        ax.set_xlabel("Threshold")
        plt.legend(["cluster"+str(i) for i in legend],loc = "upper left")
    else:
        ax.set_xlabel('Alpha')
        plt.legend(["Beta = "+str(i) for i in legend],loc = "upper left")
    plt.grid()
    plt.savefig('{}/'.format(util.get_output_dir())+category+'_threshold_bar.png')
    plt.clf()




































