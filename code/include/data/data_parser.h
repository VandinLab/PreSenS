//
// Created by X on 03/10/24.
//

#pragma once

#include <blaze/Math.h>
#include <iostream>

/**
 * Virtual class for data parser
 */
class DataParser
{

public:
    virtual ~DataParser() = default;

    /**
     * Parses the given file and converts it into a data matrix.
     */
    virtual std::shared_ptr<blaze::DynamicMatrix<double>> parse(const std::string &filePath, int is_coreset) = 0;


};



