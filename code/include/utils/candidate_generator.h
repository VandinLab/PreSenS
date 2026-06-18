//
// Created by X on 20/01/25.
//

#pragma once

#include <iostream>
#include <string>
#include <set>
#include <blaze/Math.h>
#include <clustering_utils/kmeans.h>
#include <clustering_utils/clustering.h>
#include <utils/random.h>

using centers = std::shared_ptr<blaze::DynamicMatrix<double>>;

class CandidateGenerator {

public:

    const std::set<std::string> KNOWN_GENERATORS = {"kmpp", "random", "ch", "meb"};

    bool is_known_generator(const std::string &generator) {
        return KNOWN_GENERATORS.find(generator) != KNOWN_GENERATORS.end();
    }

    CandidateGenerator(size_t random_seed);

    ~CandidateGenerator() = default;

    centers generate_candidates_kmpp(const blaze::DynamicMatrix<double> &data, int k) ;

    centers generate_candidates_random(const blaze::DynamicMatrix<double> &data, int k);

    centers generate_candidates_ch(const blaze::DynamicMatrix<double> &data, int k);

    centers generate_candidates_meb(const blaze::DynamicMatrix<double> &data, int k);

private:
    Random random_;

};

