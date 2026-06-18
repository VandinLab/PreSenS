//
// Created by X on 09/10/24.
//

#include <coresets/sensitivity_sampling.h>

using namespace coresets;

/**
 * Construct a new Sensitivity Sampling coreset object
 * @param coreset_size
 */
SensitivitySampling::SensitivitySampling(size_t coreset_size) :
        coreset_size_(coreset_size) {
    coreset_ = std::make_shared<Coreset>(coreset_size);
}

std::shared_ptr<blaze::DynamicVector<double>>
SensitivitySampling::compute_distribution_from_assignment_list(const blaze::DynamicMatrix<double> &data,
                                                               const std::vector<std::shared_ptr<Clustering>> &assignment_list) {

    auto n = data.rows();

    if (assignment_list.empty()) {
        throw std::invalid_argument("The list of assignments must not be empty");
    }

    blaze::DynamicVector<double> distribution(n, 0.0);
    int idx_iter = 0;

    for (auto &current_clustering : assignment_list) {

        auto current_assignment = current_clustering->get_assignment();

        if (assignment_list.size() == 1) {

            // -- compute the four terms of distribution
            auto k = current_assignment.get_num_clusters();
            auto clusterIndices = current_assignment.get_cluster_indices();
            double num_sum, cost_by_cluster_sum;
            double total_cost = current_assignment.get_kmeans_cost();

            std::vector<size_t> size_by_cluster; // |C_j|
            std::vector<double> cost_by_cluster; // cost(C_j, A)

            for (int idx_c = 0; idx_c < k; idx_c++) {
                auto points_by_cluster = current_assignment.get_points_by_cluster(idx_c);
                size_by_cluster.push_back(points_by_cluster->size());

                // -- compute cost(C_j, A)
                cost_by_cluster_sum = 0.0;
                for (const auto &idx_j: *points_by_cluster) {
                    cost_by_cluster_sum += current_assignment.get_distance(idx_j);
                }
                // -- add epsilon to avoid division by zero
                if (cost_by_cluster_sum == 0.0) {
                    cost_by_cluster_sum = 1e-8;
                }
                // printf("Cluster %d has %ld points\n", idx_c, points_by_cluster->size());
                // printf("Cluster %d has cost %.8f\n", idx_c, cost_by_cluster_sum);
                cost_by_cluster.push_back(cost_by_cluster_sum);
            }

            size_t n_term_1 = 0, n_term_2 = 0, n_term_3 = 0, n_term_4 = 0;

            // -- compute distribution for each point
            #pragma omp parallel for default(none) shared(distribution, size_by_cluster, cost_by_cluster, total_cost, n, k, current_assignment) reduction(+:n_term_1, n_term_2, n_term_3, n_term_4)
            for (size_t idx_p = 0; idx_p < n; idx_p++) {
                auto cluster_idx = current_assignment.get_cluster(idx_p);
                auto cost_p = current_assignment.get_distance(idx_p);
                auto cost_C_j_A = cost_by_cluster[cluster_idx];
                auto size_C_j = size_by_cluster[cluster_idx];

                auto term1 = 1.0 / ((double) (k * size_C_j)); // 1 / (k * |C_j|)
                auto term2 = (double) cost_p / ((double) (k * cost_C_j_A)); // cost(p, A) / (k * cost(C_j, A))
                auto term3 = cost_p / total_cost; // cost(p, A) / cost(P, A)
                auto term4 = cost_C_j_A / (size_C_j * total_cost); // Delta_j  / cost(P, A)


                distribution[idx_p] = (term1 + term2 + term3 + term4) * 0.25;

            }

            // -- print max terms (useful for analysis)
            // printf("Max terms: Term 1: %ld, Term 2: %ld, Term 3: %ld, Term 4: %ld\n", n_term_1, n_term_2, n_term_3, n_term_4);

        } else {

            // -- this branch is basically entered for SensS-20kmpp, which has len(assignment_list) = 20

            auto current_sensitivities = current_assignment.get_normalized_costs();
            // -- take the maximum sensitivity for each point
            int count_points = 0;
            #pragma omp parallel for default(none) shared(distribution, current_sensitivities, n) reduction(+:count_points)
            for (size_t idx_p = 0; idx_p < n; idx_p++) {
                if (distribution[idx_p] < (*current_sensitivities)[idx_p]) {
                    count_points++;
                    distribution[idx_p] = (*current_sensitivities)[idx_p];
                }
            }

            ++idx_iter;
            // printf("In iteration %d/%d, Points updated: %d\n", idx_iter, (int) assignment_list.size(), count_points);

        }
    }

    return std::make_shared<blaze::DynamicVector<double>>(distribution);
}


std::shared_ptr<blaze::DynamicVector<double>>
SensitivitySampling::compute_distribution_from_center_list(const blaze::DynamicMatrix<double> &data,
                                                            const std::vector<centers> &center_list) {

    auto n = data.rows();

    if (center_list.empty()) {
        throw std::invalid_argument("The list of centers must not be empty");
    }

    blaze::DynamicVector<double> distribution(n, 0.0);

    int idx_iter = 0;

    for (auto &current_centers: center_list) {
        // -- here, since data could be different from the data used to compute the assignment,
        // we need to re-compute it assignment
        auto k = current_centers->rows();
        ClusteringAssignment new_assignment(n, (int) k);
        new_assignment.assign_all(data, *current_centers);

        if (center_list.size() == 1) {
            // -- compute the four terms of distribution
            // -- the terms are referred with respect to assignment A (current_centers), with labels from 0 to 2*k-1
            auto clusterIndices = new_assignment.get_cluster_indices();
            double num_sum, cost_by_cluster_sum;
            double total_cost = new_assignment.get_kmeans_cost();

            std::vector<size_t> size_by_cluster; // |C_j|
            std::vector<double> cost_by_cluster; // cost(C_j, A)

            for (int idx_c = 0; idx_c < k; idx_c++) {
                auto points_by_cluster = new_assignment.get_points_by_cluster(idx_c);
                size_by_cluster.push_back(points_by_cluster->size());

                // -- compute cost(C_j, A)
                cost_by_cluster_sum = 0.0;
                for (const auto &idx_j: *points_by_cluster) {
                    cost_by_cluster_sum += new_assignment.get_distance(idx_j);
                }
                // -- add epsilon to avoid division by zero
                if (cost_by_cluster_sum == 0.0) {
                    cost_by_cluster_sum = 1e-8;
                }
                // printf("Cluster %d has %ld points\n", idx_c, points_by_cluster->size());
                // printf("Cluster %d has cost %.8f\n", idx_c, cost_by_cluster_sum);
                cost_by_cluster.push_back(cost_by_cluster_sum);
            }

            size_t n_term_1 = 0, n_term_2 = 0, n_term_3 = 0, n_term_4 = 0;

            // -- compute distribution for each point
            #pragma omp parallel for default(none) shared(distribution, size_by_cluster, cost_by_cluster, total_cost, n, k, new_assignment) reduction(+:n_term_1, n_term_2, n_term_3, n_term_4)
            for (size_t idx_p = 0; idx_p < n; idx_p++) {
                auto cluster_idx = new_assignment.get_cluster(idx_p);
                auto cost_p = new_assignment.get_distance(idx_p);
                auto cost_C_j_A = cost_by_cluster[cluster_idx];
                auto size_C_j = size_by_cluster[cluster_idx];

                auto term1 = 1.0 / ((double) (k * size_C_j)); // 1 / (k * |C_j|)
                auto term2 = (double) cost_p / ((double) (k * cost_C_j_A)); // cost(p, A) / (k * cost(C_j, A))
                auto term3 = cost_p / total_cost; // cost(p, A) / cost(P, A)
                auto term4 = cost_C_j_A / (size_C_j * total_cost); // Delta_j  / cost(P, A)

                distribution[idx_p] = (term1 + term2 + term3 + term4) * 0.25;

            }


        } else {

            auto current_sensitivities = new_assignment.get_normalized_costs();
            // -- take the maximum sensitivity for each point
            int count_points = 0;
            #pragma omp parallel for default(none) shared(distribution, current_sensitivities, n) reduction(+:count_points)
            for (size_t idx_p = 0; idx_p < n; idx_p++) {
                if (distribution[idx_p] < (*current_sensitivities)[idx_p]) {
                    count_points++;
                    distribution[idx_p] = (*current_sensitivities)[idx_p];
                }
            }

            ++idx_iter;
            // -- printf("In iteration %d/%d, Points updated: %d\n", idx_iter, (int) center_list.size(), count_points);
        }

    }

    return std::make_shared<blaze::DynamicVector<double>>(distribution);

}

centers SensitivitySampling::compute_centers(const blaze::DynamicMatrix<double> &data, int k, int n_init) {
    KMeans kmeans(k, 0, true, 1e-4, n_init);
    auto clustering = kmeans.fit(data, false);
    return std::make_shared<blaze::DynamicMatrix<double>>(clustering->get_centers());
}

std::shared_ptr<Clustering>
SensitivitySampling::compute_assignment(const blaze::DynamicMatrix<double> &data, int k, int n_init) {
    KMeans kmeans(k, 0, true, 1e-4, n_init);
    auto clustering = kmeans.fit(data, false);
    return clustering;
}


std::shared_ptr<Coreset> SensitivitySampling::get_coreset() {
    return coreset_;
}

void
SensitivitySampling::run(const std::shared_ptr<blaze::DynamicVector<double>> &distribution) {

    auto sampled_indices = random_.choice(coreset_size_, *distribution);

    // -- loop through sampled points and calculate the weight associated with each of these points
    for (size_t j = 0; j < coreset_size_; j++) {
        size_t sampled_point_index = (*sampled_indices)[j];
        auto sensitivity_j = (*distribution)[sampled_point_index];
        // printf("Sampled point %ld has sensitivity %.8f\n", sampled_point_index, sensitivity_j);
        auto weight_j = 1.0 / ((double) coreset_size_ * sensitivity_j);
        // -- add point to coreset
        coreset_->add_point(sampled_point_index, weight_j);
    }

}

