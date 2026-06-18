//
// Created by X on 03/10/24.
//

#include "data/csv_parser.h"

std::shared_ptr<blaze::DynamicMatrix<double>> CsvParser::parse(const std::string &filePath, int is_coreset) {

    if (!boost::filesystem::exists(filePath)) {
        throw std::invalid_argument("File does not exist");
    }

    std::ifstream fileStream(filePath);
    // -- unzip file if it is gzipped
    boost::iostreams::filtering_streambuf <boost::iostreams::input> filteredInputStream;
    if (boost::filesystem::path(filePath).extension() == ".gz") {
        filteredInputStream.push(boost::iostreams::gzip_decompressor());
    }
    filteredInputStream.push(fileStream);
    std::istream inData(&filteredInputStream);

    auto dataMatrix = std::make_shared < blaze::DynamicMatrix < double >> (0, 0);

    size_t nLine = 0, currRows = 0, dim = 0;
    size_t rowsAlloc = 2e7;

    while (inData.good()) {

        std::string line;
        std::getline(inData, line);
        ++nLine;

        // -- check for empty lines
        if (line.empty()) {
            continue;
        }

        std::vector<double> values;
        auto a = boost::spirit::x3::double_;
        if (boost::spirit::x3::phrase_parse(
                line.begin(), // Iterator& first
                line.end(), // Iterator last
                (boost::spirit::x3::double_ % ','), // Parser& p
                boost::spirit::x3::space, // Skipper& skipper
                values) // Attribute& attr
                ) {
            if (nLine == 1) {
                dim = values.size();
                dataMatrix->resize(rowsAlloc, dim - is_coreset);
            }

            if (values.size() != dim) {
                printf("Skipping line no %ld: expected %ld values but got %ld.\n", nLine, dim, values.size());
                continue;
            }

            if (currRows >= dataMatrix->rows()) {
                dataMatrix->resize(dataMatrix->rows() + rowsAlloc, dim - is_coreset);
            }

            // -- fill in the values
            size_t currCol = 0;
            for (size_t i = is_coreset; i < dim; ++i) {
                (*dataMatrix)(currRows, currCol) = values[i];
                ++currCol;
            }


            ++currRows;

        } else {
            std::stringstream errMsg;
            errMsg << "Parsing error at line " << nLine;
            throw std::invalid_argument(errMsg.str());
        }

    }

    // -- at the end of the file, resize the matrix to the actual number of rows
    dataMatrix->resize(currRows, dim - is_coreset);
    dataMatrix->shrinkToFit();

    return dataMatrix;

}