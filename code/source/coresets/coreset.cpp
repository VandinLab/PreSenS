//
// Created by X on 07/10/24.
//

#include <coresets/coreset.h>

Coreset::Coreset(size_t coreset_size) : coreset_size_(coreset_size), total_ins_(0) {
    points_.reserve(coreset_size);
}

long long Coreset::index_of(size_t index, bool is_ext) const {
    for (long long i = 0; i < points_.size(); i++) {
        if (points_[i].index_ == index && points_[i].is_ext_ == is_ext) {
            return i;
        }
    }
    return -1;
}

void Coreset::add_point(size_t index, double weight) {
    size_t coreset_index = index_of(index, false);
    total_ins_++;
    if (coreset_index == -1) {
        points_.emplace_back(index, weight, false);
    } else {
        // -- need to add weight
        points_[coreset_index].weight_ += weight;
    }
}

void Coreset::replace_point(size_t old_index, size_t new_index, double new_weight) {

    auto coreset_index = index_of(old_index, false);
    if (coreset_index != -1) {
        points_[coreset_index].index_ = new_index;
        points_[coreset_index].weight_ = new_weight;
    } else {
        std::cerr << "Index : << " << old_index << " not found in coreset" << std::endl;
    }
}

void Coreset::replace_point_by_index(size_t coreset_index, size_t new_index, double new_weight) {
    if (coreset_index < this->total_ins_) {
        points_[coreset_index].index_ = new_index;
        points_[coreset_index].weight_ = new_weight;
    } else {
        std::cerr << "Index : << " << coreset_index << " is larger than coreset indexes" << std::endl;
    }
}

void Coreset::add_external_point(size_t index, blaze::DynamicVector<double> &point, double weight) {
    size_t coreset_index = index_of(index, true);
    total_ins_++;
    if (coreset_index == -1) {
        points_.emplace_back(index, weight, true);
        external_points_.emplace(index, std::make_shared < blaze::DynamicVector < double >> (point));
    } else {
        // -- need to add weight
        points_[coreset_index].weight_ += weight;
    }
}

CoresetPoint Coreset::at(size_t index) const {
    return points_[index];
}

size_t Coreset::current_size() const {
    return points_.size();
}

size_t Coreset::total_points() const {
    return total_ins_;
}

std::shared_ptr <blaze::DynamicMatrix<double>>
Coreset::get_coreset_points(const blaze::DynamicMatrix<double> &original_points) const {

    auto n = original_points.rows();
    auto d = original_points.columns();

    // -- d + 1 since we need to store the weight of the point
    auto weighted_point = blaze::DynamicVector<double>(d + 1);
    std::shared_ptr <blaze::DynamicVector<double>> curr_ext;

    size_t coreset_size = points_.size();
    auto coreset = std::make_shared < blaze::DynamicMatrix < double >> (coreset_size, d + 1);

    for (size_t idx_point = 0; idx_point < coreset_size; ++idx_point) {
        auto &point = points_[idx_point];
        weighted_point.at(0) = point.weight_;
        if (point.is_ext_) {
            // -- add external point
            curr_ext = external_points_.at(point.index_);
        }

        // -- copy the point
        for (size_t i = 0; i < d; i++) {
            if (point.is_ext_) {
                weighted_point.at(i + 1) = curr_ext->at(i);
            } else {
                weighted_point.at(i + 1) = original_points(point.index_, i);
            }
        }

        blaze::row(*coreset, idx_point) = blaze::trans(weighted_point);
    }

    return coreset;

}

std::shared_ptr <Coreset>
Coreset::create_from_data(const std::shared_ptr <blaze::DynamicMatrix<double>> &data) {

    auto dplusone = data->columns();
    auto d = dplusone - 1;
    auto coreset_size = data->rows();

    auto coreset = std::make_shared<Coreset>(coreset_size);

    for (size_t i = 0; i < coreset_size; i++) {
        blaze::DynamicVector<double> point(d);
        double weight = data->at(i, 0);
        for (size_t j = 1; j < dplusone; j++) {
            point[j - 1] = data->at(i, j);
        }
        coreset->add_external_point(i, point, weight);
    }

    return coreset;

}

// TODO: use parallelism
double Coreset::compute_kmeans_cost(const blaze::DynamicMatrix<double> &centers) const {

    double kmeans_cost = 0.0;
    auto m = this->points_.size();
    auto d = centers.columns();
    int k = (int) centers.rows();
    auto coreset_points = std::make_shared < blaze::DynamicMatrix < double >> (m, d);
    // -- assumes that all points in the coreset have been already provided (not indices)
    // -- need to retrieve points in external points map
    for (size_t i = 0; i < m; i++) {
        auto point = points_[i];
        if (!point.is_ext_) {
            throw std::runtime_error("Indices points are not allowed in this method");
        }
        blaze::row(*coreset_points, i) = blaze::trans(*external_points_.at(point.index_));
    }

    // -- compute assignment of points to centers
    ClusteringAssignment assignment(m, k);
    assignment.assign_all(*coreset_points, centers);

    // -- compute weighted cost
    for (size_t i = 0; i < m; i++) {
        auto distance = assignment.get_distance(i);
        // std::cout << "Point weight: " << points_[i].weight_ << " distance: " << distance << std::endl;
        kmeans_cost += points_[i].weight_ * distance;
    }

    return kmeans_cost;


}

void Coreset::write_coreset(const blaze::DynamicMatrix<double> &original_points, std::ostream &out) const {
    auto coreset = get_coreset_points(original_points);
    auto n = coreset->rows();
    auto d = coreset->columns();
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            out << coreset->at(i, j);
            if (j < d - 1) {
                out << ",";
            }
        }
        out << "\n";
    }
}

