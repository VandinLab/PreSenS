//
// Created by X on 03/10/24.
//

#pragma once
#include <blaze/Math.h>
#include <iostream>
#include <clustering_utils/clustering.h>
#include <utils/random.h>
#include <utils/stop_watch.h>

class KMeans
{
public:

    KMeans(int n_clusters, int max_iter, bool init_kmeanspp = true, double tol = 1e-6, int n_init = 1);

    blaze::DynamicMatrix<double>
    copyRowsByIndices(const blaze::DynamicMatrix<double> &data_points, const std::vector<size_t> &indices);

    std::shared_ptr<Clustering> fit(const blaze::DynamicMatrix<double> &data_points, bool verbose = false);

    std::shared_ptr<ClusteringAssignment> initKMeansPlusPlus(const blaze::DynamicMatrix<double> &data_points);

    std::shared_ptr<std::vector<size_t>> initRandom(const blaze::DynamicMatrix<double> &data_points);

    double get_best_cost() const;

    int get_iterations_to_convergence() const;

    std::shared_ptr<blaze::DynamicMatrix<double>> get_best_centers() const;



private:
    const int n_clusters_;
    const int max_iter_;
    const double tol_;
    const int n_init_;
    const bool init_kmeanspp_;
    double best_cost_;

    int n_iter_convergence_;

    blaze::DynamicMatrix<double> best_centers_;
    ClusteringAssignment best_assignment_;

    Random random_;


};