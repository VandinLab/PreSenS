//
// Created by X on 03/10/24.
//
#include "clustering_utils/kmeans.h"

KMeans::KMeans(int n_clusters, int max_iter, bool init_kmeanspp, double tol, int n_init)
    : n_clusters_(n_clusters), max_iter_(max_iter), tol_(tol), n_init_(n_init), init_kmeanspp_(init_kmeanspp),
    best_cost_(std::numeric_limits<double>::max()), best_assignment_(0, n_clusters), n_iter_convergence_(0) {
}

blaze::DynamicMatrix<double>
KMeans::copyRowsByIndices(const blaze::DynamicMatrix<double> &data_points, const std::vector<size_t> &indices) {

    size_t k = indices.size();
    size_t d = data_points.columns();
    blaze::DynamicMatrix<double> centers(k, d);

    for (size_t idx_c = 0; idx_c < k; idx_c++) {
        blaze::row(centers, idx_c) = blaze::row(data_points, indices[idx_c]);
    }

    return centers;

}

std::shared_ptr<Clustering> KMeans::fit(const blaze::DynamicMatrix<double> &data_points, bool verbose) {

    std::cout << "Running KMeans with the following parameters:" << std::endl;
    std::cout << " - Number of clusters: " << n_clusters_ << std::endl;
    std::cout << " - Max iterations: " << max_iter_ << std::endl;
    std::cout << " - Tolerance: " << tol_ << std::endl;
    std::string init_method = init_kmeanspp_ ? "kmeans++" : "random";
    std::cout << " - Initialization: " << init_method << std::endl;
    std::cout << " - Number of initializations: " << n_init_ << std::endl;

    size_t n = data_points.rows();
    size_t d = data_points.columns();
    int k = n_clusters_;


    for (int idx_trial = 0; idx_trial < n_init_; ++idx_trial) {

        std::cout << "Running #" << (idx_trial+1) << "/" << n_init_ << " initialization..." << std::endl;

        std::vector<size_t> initial_centers_indices;
        ClusteringAssignment current_assignment(n, k);

        if (init_kmeanspp_) {
            current_assignment = *initKMeansPlusPlus(data_points);
            initial_centers_indices = *current_assignment.get_cluster_indices();
        } else {
            initial_centers_indices = *initRandom(data_points);
        }

        auto current_centers_data = copyRowsByIndices(data_points, initial_centers_indices);
        int idx_iter = 0;

        if (max_iter_ == 0) {
            current_assignment.assign_all(data_points, current_centers_data);
        }

        if (max_iter_ > 0) {

            blaze::DynamicMatrix<double> new_centers_data(k, d);

            for (idx_iter = 0; idx_iter < max_iter_; idx_iter++) {
                // -- assign points to clusters
                current_assignment.assign_all(data_points, current_centers_data);

                for (int idx_c = 0; idx_c < k; idx_c++) {
                    auto points_in_cluster = current_assignment.get_points_by_cluster(idx_c);
                    size_t num_points_in_cluster = points_in_cluster->size();
                    if (num_points_in_cluster > 0) {
                        // -- compute new centroid
                        blaze::DynamicVector<double> new_centroid(d, 0.0);
                        for (size_t idx_p = 0; idx_p < num_points_in_cluster; idx_p++) {
                            // init new_centroid to 0
                            auto cluster_point_idx = points_in_cluster->at(idx_p);
                            for (size_t idx_d = 0; idx_d < d; idx_d++) {
                                new_centroid[idx_d] += data_points.at(cluster_point_idx, idx_d);
                            }
                        }

                        new_centroid /= static_cast<double >(num_points_in_cluster);

                        blaze::row(new_centers_data, idx_c) = blaze::trans(new_centroid);
                    }

                }

                // -- compute tolerance for convergence
                auto diff_abs_matrix_centers = blaze::abs(new_centers_data - current_centers_data);
                auto diff_abs_squared = blaze::pow(diff_abs_matrix_centers, 2);
                auto frobenius_norm = blaze::sqrt(blaze::sum(diff_abs_squared));

                current_centers_data = new_centers_data;
                // -- reset new centers matrix
                new_centers_data.resize(k, d, false);

                if (frobenius_norm < tol_) {
                    if (verbose)
                        std::cout << "-- Converged after " << idx_iter << " iterations --" << std::endl;
                    break;
                }

                if (verbose) {
                    double current_cost = current_assignment.get_kmeans_cost();
                    std::cout << "Iteration " << idx_iter << ": cost = " << current_cost << std::endl;
                }


            }

        }


        if (idx_trial == 0) {
            auto cost = current_assignment.get_kmeans_cost();
            best_cost_ = cost;
            best_centers_ = current_centers_data;
            best_assignment_ = current_assignment;

        } else {
            double current_cost = current_assignment.get_kmeans_cost();
            if (current_cost < best_cost_) {
                best_cost_ = current_cost;
                best_centers_ = current_centers_data;
                best_assignment_ = current_assignment;
                n_iter_convergence_ = idx_iter;
            }
        }

    }

    std::cout << "Best k-means cost: " << best_cost_ << std::endl;

    return std::make_shared<Clustering>(best_assignment_, best_centers_);


}

std::shared_ptr<ClusteringAssignment> KMeans::initKMeansPlusPlus(const blaze::DynamicMatrix<double> &data_points) {

    size_t n = data_points.rows();
    size_t d = data_points.columns();
    int k = n_clusters_;


    auto clusters = std::make_shared<ClusteringAssignment>(n, k);

    blaze::DynamicVector<double> min_distances(n, std::numeric_limits<double>::max());

    size_t picked_center = 0;
    for (int idx_c = 0; idx_c < k; idx_c++) {
        if (idx_c == 0) {
            // -- pick first center uniformly at random
            auto gen = random_.getIndexer(n);
            picked_center = gen.next();
        } else {
            // -- compute squared distances between points and center picked at the previous iteration
            #pragma omp parallel for default(none) shared(data_points, picked_center, min_distances, n, clusters)
            for (size_t idx_p = 0; idx_p < n; idx_p++) {
                double distance = blaze::sqrNorm(blaze::row(data_points, idx_p) - blaze::row(data_points, picked_center));
                if (distance < min_distances[idx_p]) {
                    min_distances[idx_p] = distance;
                    clusters->assign(idx_p, picked_center, distance);
                }
            }

            // -- pick next center with probability proportional to squared distance
            picked_center = random_.choice(min_distances);
        }
    }

    // -- final reassignment
    #pragma omp parallel for default(none) shared(data_points, picked_center, min_distances, n, clusters)
    for (size_t idx_p = 0; idx_p < n; idx_p ++) {
        double distance = blaze::norm(blaze::row(data_points, idx_p) - blaze::row(data_points, picked_center));
        distance = distance * distance;
        if (distance < min_distances[idx_p]) {
            min_distances[idx_p] = distance;
            clusters->assign(idx_p, picked_center, distance);
        }
    }

    return clusters;

}

std::shared_ptr<std::vector<size_t>> KMeans::initRandom(const blaze::DynamicMatrix<double> &data_points) {

    size_t n = data_points.rows();
    size_t d = data_points.columns();
    int k = n_clusters_;

    auto clusters = std::make_shared<ClusteringAssignment>(n, k);
    std::vector<size_t> picked_centers_indices (k, 0.0);

    auto gen = random_.getIndexer(n);
    for (size_t idx_c = 0; idx_c < k; idx_c++) {
        while (true) {
            size_t picked_center = gen.next();
            // -- check if center has already been picked
            if (std::find(picked_centers_indices.begin(), picked_centers_indices.end(), picked_center) ==
                picked_centers_indices.end()) {
                picked_centers_indices[idx_c] = picked_center;
                break;
            }
        }

    }

    return std::make_shared<std::vector<size_t>>(picked_centers_indices);

}

int KMeans::get_iterations_to_convergence() const {
    return n_iter_convergence_;
}

double KMeans::get_best_cost() const {
    return best_cost_;
}