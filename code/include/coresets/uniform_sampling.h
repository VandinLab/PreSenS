#pragma once

#include <algorithm>
#include <vector>
#include <iostream>

#include <clustering_utils/clustering.h>
#include <coresets/coreset.h>
#include <utils/random.h>

namespace coresets
{
    class UniformSampling
    {
    public:

        UniformSampling(size_t coreset_size);

        void
        run(const blaze::DynamicMatrix<double> &data);

        std::shared_ptr<Coreset> get_coreset();

    private:
        Random random;


        const size_t coreset_size_;
        std::shared_ptr<Coreset> coreset_;

    };
}
