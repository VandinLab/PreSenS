//
// Created by X on 03/10/24.
//

#pragma once
#include <blaze/Math.h>
#include <iostream>
#include <set>
#include <omp.h>

class ClusteringAssignment
{
public:

    ClusteringAssignment(size_t num_points, int num_clusters);

    ClusteringAssignment &operator=(const ClusteringAssignment &other);

    ~ClusteringAssignment();

    void assign(size_t point_index, size_t cluster_index, double distance);

    void assign_all(const blaze::DynamicMatrix<double> &data_points, const blaze::DynamicMatrix<double> &centers);

    int get_cluster(size_t point_index) const;

    int get_num_clusters() const;

    double get_distance(size_t point_index) const;

    double get_kmeans_cost() const;

    std::shared_ptr<blaze::DynamicVector<double>> get_normalized_costs() const;

    std::shared_ptr<std::vector<size_t>> get_points_by_cluster(int cluster_index) const;

    size_t get_num_points_in_cluster(int cluster_index) const;

    std::shared_ptr<std::vector<size_t>> get_cluster_indices() const;

private:
    size_t num_points_;
    int num_clusters_;
    blaze::DynamicVector<double> distances_;
    blaze::DynamicVector<size_t> assignments_;
};

class Clustering {

public:
    Clustering(const ClusteringAssignment &assignment, const blaze::DynamicMatrix<double> &centers);
    ClusteringAssignment &get_assignment();
    blaze::DynamicMatrix<double> &get_centers();

private:
    ClusteringAssignment assignment_;
    blaze::DynamicMatrix<double> centers_;
};
