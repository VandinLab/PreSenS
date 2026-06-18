//
// Created by Cristian Boldrin on 09/10/24.
//

#pragma once

#include <iostream>
#include <vector>
#include <clustering_utils/kmeans.h>
#include <clustering_utils/clustering.h>
#include <utils/random.h>
#include <coresets/coreset.h>
#include <blaze/Math.h>

using centers = std::shared_ptr<blaze::DynamicMatrix<double>>;

namespace coresets {
    class SensitivitySampling {

    public:

        explicit SensitivitySampling(size_t coreset_size);

        ~SensitivitySampling() = default;

        static std::shared_ptr<blaze::DynamicVector<double>>
        compute_distribution_from_assignment_list(const blaze::DynamicMatrix<double> &data,
                                              const std::vector<std::shared_ptr<Clustering>> &assignment);

        // -- used for oracle
        static std::shared_ptr<blaze::DynamicVector<double>>
        compute_distribution_from_center_list(const blaze::DynamicMatrix<double> &data,
                                              const std::vector<centers> &center_list);

        void run(const std::shared_ptr<blaze::DynamicVector<double>> &distribution);

        std::shared_ptr<Coreset> get_coreset();

        static centers compute_centers(const blaze::DynamicMatrix<double> &data, int k, int n_init);

        static std::shared_ptr<Clustering> compute_assignment(const blaze::DynamicMatrix<double> &data, int k, int n_init);


    private:

        // -- random generator
        Random random_;

        // -- variable to track the internal assignment
        std::shared_ptr<ClusteringAssignment> assignment_;

        // -- coreset
        const size_t coreset_size_;
        std::shared_ptr<Coreset> coreset_;


    };
}
