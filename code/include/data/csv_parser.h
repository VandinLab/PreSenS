//
// Created by X on 03/10/24.
//

#pragma once

#include "data/data_parser.h"

#include <fstream>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/spirit/home/x3.hpp>
#include <omp.h>

class CsvParser : public DataParser
{
public:

    std::shared_ptr<blaze::DynamicMatrix<double>> parse(const std::string &filePath, int is_coreset) override;



};

