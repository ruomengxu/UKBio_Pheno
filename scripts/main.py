import etl
import clustering
import visualization

def main():
    eids, sparse_feature_matrix, feature_names = etl.load_sparse_feature_matrix()
    feature_matrix_after_feature_selection, selected_indices =  clustering.feature_selection(sparse_feature_matrix, .9995)
    W, H, reconstruction_err = clustering.get_nmf_w_h(feature_matrix_after_feature_selection, n_components=5)
    labels = clustering.get_cluster_assignments(eids, W)
    visualization.nonzero_threshold_plot(H,feature_names,selected_indices)
    visualization.percentage_feature_in_cluster_vs_cohort('non-cancer-illness', labels, H, feature_names, sparse_feature_matrix.shape[0])
    visualization.silhouette_plot(eids, feature_matrix_after_feature_selection, distance_metric='cosine')
    visualization.iteration_vs_error(feature_matrix_after_feature_selection, 4)
    visualization.vis_grs(labels)
    visualization.vis_feature_bar(sparse_feature_matrix)
    # clustering.hyperparameter_tuning(feature_matrix_after_feature_selection)
    # feature_matrix_reduced = clustering.dim_reduction(sparse_feature_matrix)
    # optimal_k = clustering.get_optimal_k(eids, sparse_feature_matrix, distance_metric='cosine')
    visualization.vis_info_table(H,feature_names)
    visualization.nonzero_alpha_beta_plot(4)

if __name__ == '__main__':
    main()