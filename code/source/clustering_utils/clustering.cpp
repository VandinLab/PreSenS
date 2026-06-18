//
// Created by X on 03/10/24.
//

#include "clustering_utils/clustering.h"

ClusteringAssignment::ClusteringAssignment(size_t num_points, int num_clusters) : num_points_(num_points),
                                                                                  num_clusters_(num_clusters), distances_(num_points), assignments_(num_points)
{
}

ClusteringAssignment::~ClusteringAssignment() = default;

ClusteringAssignment &ClusteringAssignment::operator=(const ClusteringAssignment &other) {
    num_points_ = other.num_points_;
    num_clusters_ = other.num_clusters_;
    distances_ = other.distances_;
    assignments_ = other.assignments_;
    return *this;
}

void ClusteringAssignment::assign(size_t point_index, size_t cluster_index, double distance) {
    if (point_index >= num_points_) {
        std::cerr << "Assign | Invalid point index: " << point_index << std::endl;
        return;
    }
    distances_[point_index] = distance;
    assignments_[point_index] = cluster_index;
}

void ClusteringAssignment::assign_all(const blaze::DynamicMatrix<double> &data_points,
                                      const blaze::DynamicMatrix<double> &centers) {
    // -- for each point, compute distances and find the closest center
    #pragma omp parallel for default(none) shared(data_points, centers)
    for (size_t idx_p = 0; idx_p < num_points_; ++idx_p) {
        double min_sqr_distance = std::numeric_limits<double>::max();
        int point_label = -1;

        auto point_row = blaze::row(data_points, idx_p);
        // -- could be parallelized, but we consider k in [10, 20, 30, 50], so it may not be worth it due to overhead
        for (int idx_c = 0; idx_c < num_clusters_; ++idx_c) {
            // -- compute squared L2 norm (for efficiency) between point and center
            double sqr_distance = blaze::sqrNorm(point_row - blaze::row(centers, idx_c));

            if (sqr_distance < min_sqr_distance) {
                min_sqr_distance = sqr_distance;
                point_label = idx_c;
            }
        }

        // -- assign the point to the closest center (thread safe for idx_p)
        assign(idx_p, point_label, min_sqr_distance);
    }
}

int ClusteringAssignment::get_num_clusters() const {
    return num_clusters_;
}

double ClusteringAssignment::get_distance(size_t point_index) const {
    if (point_index >= num_points_) {
        std::cerr << "Invalid point index: " << point_index << std::endl;
        return -1;
    }
    return distances_[point_index];
}

double ClusteringAssignment::get_kmeans_cost() const {
    double cost = 0.0;
    #pragma omp parallel for default(none) reduction(+:cost)
    for (size_t idx_p = 0; idx_p < num_points_; idx_p++) {
        cost += distances_[idx_p];
    }
    return cost;
}


std::shared_ptr<blaze::DynamicVector<double>> ClusteringAssignment::get_normalized_costs() const {
    auto normalized_costs = std::make_shared<blaze::DynamicVector<double>>(num_points_);
    double total_cost = get_kmeans_cost();
    #pragma omp parallel for default(none) shared(normalized_costs, total_cost)
    for (size_t idx_p = 0; idx_p < num_points_; idx_p++) {
        (*normalized_costs)[idx_p] = distances_[idx_p] / total_cost;
    }
    return normalized_costs;
}

std::shared_ptr<std::vector<size_t>> ClusteringAssignment::get_points_by_cluster(int cluster_index) const {

    auto point_by_cluster = std::make_shared<std::vector<size_t>>();
    for (size_t idx_p = 0; idx_p < num_points_; idx_p++) {
        if (assignments_[idx_p] == cluster_index) {
            point_by_cluster->push_back(idx_p);
        }
    }

    return point_by_cluster;

}

size_t ClusteringAssignment::get_num_points_in_cluster(int cluster_index) const {
    size_t count = 0;
    for (size_t idx_p = 0; idx_p < num_points_; idx_p++) {
        if (assignments_[idx_p] == cluster_index) {
            count++;
        }
    }
    return count;
}

int ClusteringAssignment::get_cluster(size_t point_index) const {
    if (point_index >= num_points_) {
        std::cerr << "Invalid point index: " << point_index << std::endl;
        return -1;
    }
    return assignments_[point_index];
}


std::shared_ptr<std::vector<size_t>> ClusteringAssignment::get_cluster_indices() const {
    // -- create unique labels
    std::set<size_t> unique_labels;
    for (size_t idx_p = 0; idx_p < num_points_; idx_p++) {
        unique_labels.insert(assignments_[idx_p]);
    }

    return std::make_shared<std::vector<size_t>>(unique_labels.begin(), unique_labels.end());
}

// --- Clustering class ---

/**
 * Clustering object constructor
 * @param assignment
 * @param centers
 */
Clustering::Clustering(const ClusteringAssignment &assignment, const blaze::DynamicMatrix<double> &centers) :
        assignment_(assignment), centers_(centers)
{
}

ClusteringAssignment &Clustering::get_assignment() {
    return assignment_;
}

blaze::DynamicMatrix<double> &Clustering::get_centers() {
    return centers_;
}