//
// Created by X on 03/10/24.
//
#include <utils/random.h>

static std::mt19937 randomEngine;

RandomIndexer::RandomIndexer(size_t s) : sampler(0, s - 1)
{
}

size_t
RandomIndexer::next()
{
    return sampler(randomEngine);
}

void
Random::initialize(size_t fixedSeed)
{
    if (fixedSeed == 0)
    {
        // Source: https://stackoverflow.com/questions/15509270/does-stdmt19937-require-warmup
        std::array<int, 624> seedData;
        std::random_device randomDevice;
        std::generate_n(seedData.data(), seedData.size(), std::ref(randomDevice));
        std::seed_seq randomSeq(std::begin(seedData), std::end(seedData));
        randomEngine.seed(randomSeq);
    }
    else
    {
        randomEngine.seed(static_cast<uint>(fixedSeed));
    }
}

RandomIndexer
Random::getIndexer(size_t size)
{
    return RandomIndexer(size);
}

double
Random::getDouble()
{
    return pickRandomValue(randomEngine);
}

double Random::getDouble(double min, double max)
{
    std::uniform_real_distribution<double> distribution(min, max);
    return distribution(randomEngine);
}

Random::Random()
{
}

std::shared_ptr<blaze::DynamicVector<size_t>>
Random::choice(const size_t k, blaze::DynamicVector<double> &weights)
{
    auto result = std::make_shared<blaze::DynamicVector<size_t>>(k);
    result->reset();

    std::discrete_distribution<size_t> weightedChoice(weights.begin(), weights.end());

    for (size_t i = 0; i < k; i++)
    {
        size_t pickedIndex = weightedChoice(randomEngine);
        (*result)[i] = pickedIndex;
    }

    return result;
}

size_t
Random::choice(blaze::DynamicVector<double> &weights)
{
    std::discrete_distribution<size_t> weightedChoice(weights.begin(), weights.end());
    size_t pickedIndex = weightedChoice(randomEngine);
    return pickedIndex;
}

size_t
Random::stochasticRounding(double value)
{
    auto valueHigh = floor(value);
    auto valueLow = ceil(value);
    auto proba = (value - valueLow) / (valueHigh - valueLow);
    auto randomVal = this->getDouble();
    if (randomVal < proba)
    {
        return static_cast<size_t>(round(valueHigh)); // Round up
    }
    return static_cast<size_t>(round(valueLow)); // Round down
}

void Random::normal(blaze::DynamicVector<double> &vector)
{
    std::normal_distribution<double> distribution(0.0, 1.0);
    auto entryGenerator = [&distribution]() {
        return distribution(randomEngine);
    };
    std::generate(vector.begin(), vector.end(), entryGenerator);
}

double Random::get_normal_value(double mean, double stddev) {
    std::normal_distribution<double> distribution(mean, stddev);
    return distribution(randomEngine);
}

blaze::DynamicVector<double>
Random::multivariate_normal(const blaze::DynamicVector<double>& mean,
                            const blaze::DynamicMatrix<double>& cov)
{
    const size_t d = mean.size();

    if (cov.rows() != d || cov.columns() != d) {
        throw std::invalid_argument("Covariance matrix has incompatible dimensions");
    }

    // Optional but useful sanity check
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            if (std::abs(cov(i, j) - cov(j, i)) > 1e-10) {
                throw std::invalid_argument("Covariance matrix must be symmetric");
            }
        }
    }

    // Copy covariance because we will build the Cholesky factor into L
    blaze::DynamicMatrix<double> L(d, d, 0.0);

    // Manual Cholesky decomposition: cov = L * trans(L)
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = cov(i, j);

            for (size_t k = 0; k < j; ++k) {
                sum -= L(i, k) * L(j, k);
            }

            if (i == j) {
                if (sum <= 0.0) {
                    throw std::runtime_error(
                            "Covariance matrix is not positive definite");
                }
                L(i, j) = std::sqrt(sum);
            } else {
                L(i, j) = sum / L(j, j);
            }
        }
    }

    // Standard normal vector z ~ N(0, I)
    std::normal_distribution<double> dist(0.0, 1.0);
    blaze::DynamicVector<double> z(d);

    for (size_t i = 0; i < d; ++i) {
        z[i] = dist(randomEngine);
    }

    // x = mean + L * z
    return mean + L * z;
}