#include <coresets/uniform_sampling.h>

using namespace coresets;

UniformSampling::UniformSampling(size_t coreset_size) : coreset_size_(coreset_size)
{
    coreset_ = std::make_shared<Coreset>(coreset_size);
}

void
UniformSampling::run(const blaze::DynamicMatrix<double> &data)
{

    auto n = data.rows();
    // -- create a blaze::DynamicVector<double> of size n, with each element being 1/n (uniform distro)
    blaze::DynamicVector<double> sampling_distribution(n);
    sampling_distribution = 1.0 / static_cast<double>(n);

    auto sampled_indices = random.choice(coreset_size_, sampling_distribution);

    double m_double = static_cast<double>(coreset_size_);

    // Loop through the sampled points and calculate
    // the weight associated with each of these points.
    for (size_t j = 0; j < coreset_size_; j++)
    {
        size_t sampledPointIndex = (*sampled_indices)[j];

        double weight = (1.0 / (m_double / static_cast<double>(n)));

        coreset_->add_point(sampledPointIndex, weight);

        // printf("Sampled point %3ld gets weight %.5f \n", sampledPointIndex, weight);
    }

}

std::shared_ptr<Coreset> UniformSampling::get_coreset()
{
    return coreset_;
}
