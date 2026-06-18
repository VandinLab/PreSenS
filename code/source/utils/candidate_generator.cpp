//
// Created by X on 20/01/25.
//

#include <utils/candidate_generator.h>

CandidateGenerator::CandidateGenerator(size_t random_seed) {
    Random::initialize(random_seed);
}


centers CandidateGenerator::generate_candidates_kmpp(const blaze::DynamicMatrix<double> &data, int k) {
    // -- run kmpp
    KMeans kmeans(k, 0, true, 1e-4, 1);
    auto clustering = kmeans.fit(data, false);
    // -- update generated centers
    return std::make_shared<blaze::DynamicMatrix<double>>(clustering->get_centers());
}

centers CandidateGenerator::generate_candidates_random(const blaze::DynamicMatrix<double> &data, int k) {
    // -- run random initialization (flag = 1)
    KMeans kmeans(k, 0, false, 1e-4, 1);
    auto clustering = kmeans.fit(data, false);
    // -- update generated centers
    return std::make_shared<blaze::DynamicMatrix<double>>(clustering->get_centers());

}

/**
 * Generate k random points within the convex hull of the data.
 *
 * Note that points are not place uniformly at random. It is more likely to generate points in dense areas of data.
 * @param data
 * @param k
 * @return generated centers
 */
centers CandidateGenerator::generate_candidates_ch(const blaze::DynamicMatrix<double> &data, int k) {

    auto n = data.rows();
    auto d = data.columns();

    blaze::DynamicMatrix<double> generated_centers(k, d, 0);

    // -- create a range from 0 to n-1 as std::vector
    std::vector<size_t> n_range(n);
    std::iota(n_range.begin(), n_range.end(), 0);

    size_t curr_data_samples = 2;

    for (int i = 0; i < k; i++) {
        // -- generate a random vector in [0, 1) of shape curr_data_samples x 1
        blaze::DynamicVector<double> random_vector(curr_data_samples, 1);

        double L1_norm = 0.0;
        for (size_t j = 0; j < curr_data_samples; j++) {
            auto gen = random_.getDouble();
            random_vector.at(j) = gen;
            L1_norm += gen;
        }

        // -- normalize the random vector
        blaze::DynamicVector<double> proba_vector = random_vector / L1_norm;

        std::vector<size_t> sampled_indices;

        if (curr_data_samples == n)
            sampled_indices = n_range;
        else
            sampled_indices = random_.choice(n_range, curr_data_samples);

        // -- generate a new point by taking the convex combination of the selected input points
        // -- take the dot product between proba_vector and data[sampled_indices]
        blaze::DynamicVector<double> new_center(d, 0);
        for (size_t l = 0; l < d; l++) {
            for (size_t j = 0; j < curr_data_samples; j++) {
                new_center[l] += proba_vector[j] * data(sampled_indices[j], l);
            }
        }


        // -- add the new center to the generated centers
        blaze::row(generated_centers, i) = blaze::trans(new_center);

    }

    // -- compute costs (just for visualization)
    ClusteringAssignment assignment(n, k);
    assignment.assign_all(data, generated_centers);
    auto kmeans_cost = assignment.get_kmeans_cost();
    std::cout << "KMeans cost for CH generated centers: " << kmeans_cost << std::endl;

    return std::make_shared<blaze::DynamicMatrix<double>>(generated_centers);

}

centers CandidateGenerator::generate_candidates_meb(const blaze::DynamicMatrix<double> &data, int k) {

    auto n = data.rows();
    auto d = data.columns();

    // -- Step 1: compute minimum enclosing ball
    // -- Implements algorithm from http://cm.bell-labs.co/who/clarkson/coresets2.pdf

    const int n_iter = 100; // -- corresponds to 1/\epsilon**2 in the note (Sec. 3)
    std::array<size_t, n_iter> explored_points_indices{};

    // -- randomly pick a point
    auto n_indexer = random_.getIndexer(n);
    auto initial_point_index = n_indexer.next();

    int n_explored_points = 0;
    explored_points_indices[n_explored_points] = initial_point_index;

    blaze::DynamicVector<double> starting_center = blaze::trans(blaze::row(data, initial_point_index));
    for (n_explored_points = 1; n_explored_points < n_iter; n_explored_points++) {
        // -- pick the point that is farthest from the current center
        double max_distance = 0.0;
        size_t farthest_point_index = 0;
        for (size_t j = 0; j < n; j++) {
            blaze::DynamicVector<double> current_point = blaze::trans(blaze::row(data, j));
            double current_distance = blaze::l2Norm(current_point - starting_center);
            if (current_distance > max_distance) {
                max_distance = current_distance;
                farthest_point_index = j;
            }
        }

        explored_points_indices[n_explored_points] = farthest_point_index;

        // -- move starting_center toward the farthest point
        blaze::DynamicVector<double> farthest_point = blaze::trans(blaze::row(data, farthest_point_index));
        starting_center = starting_center + (farthest_point - starting_center) / (n_explored_points + 1);
    }

    // -- compute radius
    double radius = 0.0;
    std::set<size_t> explored_points_indices_set(explored_points_indices.begin(), explored_points_indices.end());
    // -- radius is the maximum pairwise distance between starting_center and explored_point
    auto n_unique = explored_points_indices_set.size();
    for (size_t i = 0; i < n_unique; i++) {
        blaze::DynamicVector<double> current_explored_point = blaze::trans(blaze::row(data, explored_points_indices[i]));
        double current_distance = blaze::l2Norm(current_explored_point - starting_center);
        if (current_distance > radius) {
            radius = current_distance;
        }
    }

    blaze::DynamicMatrix<double> generated_centers(k, d, 0);

    // -- Step 2: generate k-arbitrary points within minimum enclosing ball
    // -- generate k vectors from ~N(center[i], 1.0) for each coordinate i
    blaze::DynamicMatrix<double> identity_matrix(d, d, 0.0);
    for (size_t i = 0; i < d; i++) {
        identity_matrix(i, i) = 1.0;
    }

    for (int j = 0; j < k; j++) {
        // -- generate random multivariate normal vector: for each entry i, sample from N(starting_point[i], 1)
        // -- create identity blaze matrix

        // -- generate random vector from multivariate normal distribution
        blaze::DynamicVector<double> random_vector = random_.multivariate_normal(starting_center, identity_matrix);

        // -- generate a length, i.e., uniform value in [1e-5, radius]
        double length = random_.getDouble(1e-5, radius);

        // -- random vector becomes length * (random_vector / ||random_vector||)
        random_vector = length * (random_vector / blaze::l2Norm(random_vector));
        blaze::row(generated_centers, j) = blaze::trans(random_vector);
    }

    // std::cout << "Returning centers" << std::endl;
    // -- compute costs (just for visualization)
    ClusteringAssignment assignment(n, k);
    assignment.assign_all(data, generated_centers);
    auto kmeans_cost = assignment.get_kmeans_cost();
    std::cout << "KMeans cost for MEB generated centers: " << kmeans_cost << std::endl;

    return std::make_shared<blaze::DynamicMatrix<double>>(generated_centers);

}