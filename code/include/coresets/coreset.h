//
// Created by X on 07/10/24.
//

#pragma once

#include <iostream>
#include <vector>
#include <blaze/Math.h>
#include <map>

#include <clustering_utils/clustering.h>

struct CoresetPoint {
    // -- index of the point in the original dataset
    size_t index_;
    // -- weight of the point
    double weight_;
    // -- flag to indicate if the point is not a point of original data
    const bool is_ext_;

    CoresetPoint(size_t index, double weight, bool is_ext) : index_(index), weight_(weight), is_ext_(is_ext) {}
};

class Coreset {

public:
    Coreset(size_t coreset_size);

    static std::shared_ptr<Coreset> create_from_data(const std::shared_ptr<blaze::DynamicMatrix<double>>& data);

    void add_point(size_t index, double weight);

    void add_external_point(size_t index, blaze::DynamicVector<double> &point, double weight);

    size_t current_size() const;

    size_t total_points() const;

    long long index_of(size_t index, bool is_center) const;

    CoresetPoint at(size_t index) const;

    void replace_point(size_t old_index, size_t new_index, double new_weight);

    void replace_point_by_index(size_t coreset_index, size_t new_index, double new_weight);

    std::shared_ptr<blaze::DynamicMatrix<double>> get_coreset_points(const blaze::DynamicMatrix<double> &original_points) const;

    double compute_kmeans_cost(const blaze::DynamicMatrix<double> &centers) const;

    void write_coreset(const blaze::DynamicMatrix<double> &original_points, std::ostream &out) const;



private:
    const size_t coreset_size_;
    std::vector<CoresetPoint> points_;
    std::map<size_t, std::shared_ptr<blaze::DynamicVector<double>>> external_points_;
    size_t total_ins_;
};
