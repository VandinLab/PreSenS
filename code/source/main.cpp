#include <iostream>
#include <iomanip>
#include <filesystem>
#include <data/data_parser.h>
#include <data/csv_parser.h>
#include <clustering_utils/kmeans.h>
#include <coresets/uniform_sampling.h>
#include <coresets/sensitivity_sampling.h>
#include <boost/algorithm/string.hpp>
#include <bits/stdc++.h>
#include <utils/candidate_generator.h>

#define OMP_NUM_THREADS 1

// -- Set of KNOWN datasets
static const std::set<std::string> KNOWN_DATASETS_CSV = {"taxi", "twitter", "NYCTaxiTrip", "NYC_YT", "IntelLab",
                                                         "coreset"};

static const std::set<std::string> KNOWN_ALGOS = {"sensitivity-sampling", "uniform-sampling"};

bool match_dataset(const std::string &dataset_name) {
    // -- check for matches of type taxi_snap001 <-> taxi, only at beginning
    for (const auto &known_dataset: KNOWN_DATASETS_CSV) {
        if (boost::starts_with(dataset_name, known_dataset)) {
            return true;
        }
    }
    return false;
}


void write_coreset_to_stream(const std::shared_ptr<blaze::DynamicMatrix < double>>

&original_data,
const std::shared_ptr<Coreset> &coreset,
const std::string &output_dir
) {

std::string output_path = output_dir + "/coreset_data.txt.gz";

namespace io = boost::iostreams;

std::ofstream out(output_path, std::ios_base::out | std::ios_base::binary);
io::filtering_streambuf<io::output> outbuf;
outbuf.
push(io::gzip_compressor(io::gzip_params(io::gzip::best_compression))
);
outbuf.
push(out);

std::cout << "Writing coreset to " << output_path << "..." <<
std::endl;
std::ostream out_stream(&outbuf);
coreset->
write_coreset(*original_data, out_stream
);

}

void write_kmeans_results(const std::shared_ptr<blaze::DynamicMatrix < double>>

&centers,
double best_cost,
int iter_to_convergence,
double time,
double time_to_parse_data,
int is_coreset,
int idx_gen,
const std::string &output_path
) {

// -- Create a stringstream for formatted output
std::stringstream ss;
ss << std::setw(2) << std::setfill('0') <<
idx_gen;
// Construct the full output path with formatted idx_gen
std::string data = is_coreset ? "coreset" : "dataset";
std::string output_path_centers = output_path + "/" + ss.str() + "_kmeans_" + data + "_centers.txt.gz";
std::string output_path_stats = output_path + "/" + ss.str() + "_kmeans_" + data + "_stats.txt";


namespace io = boost::iostreams;

std::ofstream out(output_path_centers, std::ios_base::out | std::ios_base::binary);
io::filtering_streambuf<io::output> outbuf;
outbuf.
push(io::gzip_compressor(io::gzip_params(io::gzip::best_compression))
);
outbuf.
push(out);

// -- write centers
std::ostream out_stream(&outbuf);
auto k = centers->rows();
auto d = centers->columns();
// -- print size
printf("Writing Centers of shape: %lu x %lu\n", k, d);

for (
size_t i = 0;
i<k;
i++) {
for (
size_t j = 0;
j<d;
j++) {
out_stream << centers->
at(i, j
);
if (j<d - 1) {
out_stream << ",";
}
}
out_stream << "\n";
}

// -- write stats
std::ofstream out_stats(output_path_stats);
out_stats << "Best_cost,Iter_to_convergence,Time_to_kmeans,Time_to_parse_data\n";
out_stats << std::fixed << best_cost << "," << iter_to_convergence << "," << time << "," <<
time_to_parse_data;
out_stats.

close();

}


void write_assignment_centers(const std::shared_ptr<blaze::DynamicMatrix < double>>

&centers,
const std::string &output_path
) {
std::ofstream out(output_path);
std::ofstream out_stream(output_path);

// set precision
out_stream << std::fixed << std::setprecision(3);

auto k = centers->rows();
auto d = centers->columns();

for (
size_t i = 0;
i<k;
i++) {
for (
size_t j = 0;
j<d;
j++) {
double value = centers->at(i, j);
out_stream <<
value;
if (j<d - 1)
out_stream << ",";
}
out_stream << "\n";
}

// -- flush and close the streams
out_stream.

close();

}

double parse_data(const std::string &dataset_path, std::shared_ptr<DataParser> &dataParser,
                  std::shared_ptr<blaze::DynamicMatrix < double>>

&data,
int is_coreset = 0
) {

std::cout << "Parsing data from " << dataset_path << "..." <<
std::endl;
auto start = std::chrono::high_resolution_clock::now();
if (is_coreset == 1) {
std::cout << "Data to be parsed is a coreset." <<
std::endl;
data = dataParser->parse(dataset_path, 1);
} else {
data = dataParser->parse(dataset_path, 0);
}

auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> elapsed = end - start;
printf("Data parsed in %.3f seconds.\n", elapsed.

count()

);
// -- data shape
std::cout << "Data shape: " << data->

rows()

<< " x " << data->

columns()

<<
std::endl;
return elapsed.

count();

}

centers generate_candidates(const std::string &generator, const blaze::DynamicMatrix<double> &data, int k,
                            size_t random_seed) {
    CandidateGenerator candidateGenerator(random_seed);
    if (generator == "kmpp") {
        return candidateGenerator.generate_candidates_kmpp(data, k);
    } else if (generator == "random") {
        return candidateGenerator.generate_candidates_random(data, k);
    } else if (generator == "ch") {
        return candidateGenerator.generate_candidates_ch(data, k);
    } else if (generator == "meb") {
        return candidateGenerator.generate_candidates_meb(data, k);
    } else {
        throw std::invalid_argument("Unknown generator: " + generator);
    }
}

/**
 * Get the base name of the executable
 * @param s the string to split
 * @return the name of the executable
 */
char *base_name(char *s) {
    char *start;
    if ((start = strrchr(s, '/')) == nullptr) {
        start = s;
    } else {
        ++start;
    }

    return start;
}

int main(int argc, char **argv) {

    // -- set omp threads
    omp_set_dynamic(0);
    omp_set_num_threads(OMP_NUM_THREADS);

    // -- print number of threads
    printf("Number of threads: %d\n", omp_get_max_threads());

    char *project_name = base_name(argv[0]);

    if (strcmp(project_name, "CoresetWithPredictions") == 0) {

        if (argc < 8) {
            printf("Invalid number of arguments.\n");
            printf("Usage: dataset dataset_path coreset_algorithm m k random_seed output_dir "
                   "predictor_flag predictor_path n_init\n");
            std::cout << "\t- dataset =                 dataset name" << std::endl;
            std::cout << "\t- dataset_path =            path to the dataset (csv file)" << std::endl;
            std::cout << "\t- coreset_algorithm =       coreset algorithm" << std::endl;
            std::cout << "\t- m =                       coreset size" << std::endl;
            std::cout << "\t- k =                       number of clusters" << std::endl;
            std::cout << "\t- random_seed =             random seed for reproducibility" << std::endl;
            std::cout << "\t- output_dir =              output directory" << std::endl;
            std::cout << "\t- predictor_flag =             W if you want to write the predictor table, "
                         "C if you want to compute the predictor table, "
                         "R if you want to read the predictor table" << std::endl;
            std::cout << "\t- predictor_path =             path to the predictor table (csv file)" << std::endl;
            std::cout << "\t- n_times_kmpp =            number of times (= number of sets of center) for which "
                         "computing kmpp" << std::endl;

            return 1;
        }

        const char *dataset_name(argv[1]);
        std::string dataset_path(argv[2]);
        std::string coreset_algorithm(argv[3]);
        int m = std::stoi(argv[4]);
        int k = std::stoi(argv[5]);
        size_t random_seed = std::stol(argv[6]);
        std::string output_dir(argv[7]);

        // -- ensure output dir
        if (!boost::filesystem::exists(output_dir)) {
            std::cout << "Creating output directory: " << output_dir << std::endl;
            boost::filesystem::create_directories(output_dir);
        }

        printf("Running with the following parameters:\n");
        printf(" - Dataset: %s\n", dataset_name);
        printf(" - Dataset path: %s\n", dataset_path.c_str());
        printf(" - Coreset algorithm: %s\n", coreset_algorithm.c_str());
        printf(" - Coreset size: %d\n", m);
        printf(" - Number of clusters: %d\n", k);
        printf(" - Random seed: %ld\n", random_seed);
        printf(" - Output directory: %s\n", output_dir.c_str());


        // -- check dataset
        std::shared_ptr<DataParser> dataParser;
        if (match_dataset(dataset_name)) dataParser = std::make_shared<CsvParser>();
        else {
            printf("Unknown dataset: %s\n", dataset_name);
            return 1;
        }

        // -- check algo
        if (KNOWN_ALGOS.find(coreset_algorithm) == KNOWN_ALGOS.end()) {
            printf("Unknown coreset algorithm: %s\n", coreset_algorithm.c_str());
            return 1;
        }

        // -- seed code
        Random::initialize(random_seed);

        std::shared_ptr<blaze::DynamicMatrix < double>>
        data;
        auto time_to_parse_data = parse_data(dataset_path, dataParser, data);

        // -- algorithm
        auto coreset = std::make_shared<Coreset>(m);
        StopWatch coreset_sw(false);

        if (coreset_algorithm == "uniform-sampling") {

            // -- uniform sampling
            coreset_sw.start();
            coresets::UniformSampling uniform_sampling_coreset(m);

            uniform_sampling_coreset.run(*data);
            coreset = uniform_sampling_coreset.get_coreset();

            auto time_elapsed_coreset = coreset_sw.elapsedSeconds();
            printf("Coreset created with %lu rows.\n", coreset->current_size());
            printf("Coreset elapsed time: %.3f s\n", time_elapsed_coreset);
            // -- write coreset
            write_coreset_to_stream(data, coreset, output_dir);

            // -- write times
            std::string output_path_times = output_dir + "/times.txt";
            std::ofstream out_times(output_path_times);
            out_times << "Time_to_parse_data,Time_to_predictor,Time_to_coreset\n";
            out_times << time_to_parse_data << "," << 0.0 << "," << time_elapsed_coreset;
            out_times.close();


        } else if (coreset_algorithm == "sensitivity-sampling") {

            // -- assert argc is not < 11
            if (argc < 11) {
                printf("Invalid number of arguments for sensitivity-sampling algorithm.\n");
                printf("Usage: dataset dataset_path coreset_algorithm m k random_seed output_dir "
                       "predictor_flag predictor_path n_init\n");
                std::cout << "\t- dataset =                 dataset name" << std::endl;
                std::cout << "\t- dataset_path =            path to the dataset (csv file)" << std::endl;
                std::cout << "\t- coreset_algorithm =       coreset algorithm" << std::endl;
                std::cout << "\t- m =                       coreset size" << std::endl;
                std::cout << "\t- k =                       number of clusters" << std::endl;
                std::cout << "\t- random_seed =             random seed for reproducibility" << std::endl;
                std::cout << "\t- output_dir =              output directory" << std::endl;
                std::cout << "\t- predictor_flag =             W if you want to write the predictor table, "
                             "C if you want to compute the predictor table, "
                             "R if you want to read the predictor table" << std::endl;
                std::cout << "\t- predictor_path =             path to the predictor table (csv file)" << std::endl;
                std::cout << "\t- n_times_kmpp =            number of times (= number of sets of center) for which "
                             "computing kmpp" << std::endl;

                return 1;
            }

            std::string predictor_flag(argv[8]);
            std::string predictor_path(argv[9]);

            printf(" - Predictor flag: %s\n", predictor_flag.c_str());
            printf(" - Predictor path: %s\n", predictor_path.c_str());

            // -- StopWatch for predictor centers
            double elapsed_time_predictor;
            StopWatch predictor_sw(true);

            int n_times_kmpp = std::stoi(argv[10]);

            if (predictor_flag == "R") {
                // -- PreSens
                std::vector<centers> center_list;
                auto predictor_centers = std::make_shared<blaze::DynamicMatrix < double>>();
                // -- read the predictor from file
                parse_data(predictor_path, dataParser, predictor_centers);
                elapsed_time_predictor = predictor_sw.elapsedSeconds();
                std::cout << "Predictor table read from " << predictor_path << " in " << elapsed_time_predictor << " seconds."
                          << std::endl;

                center_list.push_back(predictor_centers);
                coreset_sw.start();
                coresets::SensitivitySampling sensitivity_sampling_coreset(m);
                auto distro = coresets::SensitivitySampling::compute_distribution_from_center_list(*data, center_list);
                sensitivity_sampling_coreset.run(distro);
                coreset = sensitivity_sampling_coreset.get_coreset();

            } else {
                // -- if n_init = 20, then S-20kmpp algo is run (for using exactly k centers, must pass k/2 in input)
                std::vector<std::shared_ptr<Clustering>> assignment_list;
                printf(" - Number of times for which computing kmpp: %d\n", n_times_kmpp);
                for (int i = 0; i < n_times_kmpp; i++) {
                    std::cout << "---------- kmpp(2k) assignment " << i + 1 << "/" << n_times_kmpp
                              << " ---------- " << std::endl;
                    // -- compute O(1, 2k) bi-criteria assignment
                    auto current_assignment = coresets::SensitivitySampling::compute_assignment(*data, 2 * k, 1);
                    assignment_list.push_back(current_assignment);
                }

                elapsed_time_predictor = predictor_sw.elapsedSeconds();
                // -- write centers
                std::cout << "Predictor table computed in " << elapsed_time_predictor << " seconds." << std::endl;
                if (predictor_flag == "W" and n_times_kmpp == 1) {
                    // -- write assignment centers to file
                    write_assignment_centers(std::make_shared<blaze::DynamicMatrix < double>>(assignment_list[0]->get_centers()), predictor_path);
                }

                coreset_sw.start();
                coresets::SensitivitySampling sensitivity_sampling_coreset(m);
                auto distro = coresets::SensitivitySampling::compute_distribution_from_assignment_list(*data,
                                                                                                       assignment_list);
                sensitivity_sampling_coreset.run(distro);
                coreset = sensitivity_sampling_coreset.get_coreset();

            }


            auto time_elapsed_coreset = coreset_sw.elapsedSeconds();
            printf("Coreset created with %lu rows.\n", coreset->current_size());
            printf("Coreset elapsed time: %.3f s\n", time_elapsed_coreset);
            // -- write coreset
            write_coreset_to_stream(data, coreset, output_dir);

            // -- write times
            std::string output_path_times = output_dir + "/times.txt";
            std::ofstream out_times(output_path_times);
            out_times << "Time_to_parse_data,Time_to_predictor,Time_to_coreset\n";
            out_times << time_to_parse_data << "," << elapsed_time_predictor << "," << time_elapsed_coreset;
            out_times.close();

        } else {
            printf("Unknown coreset algorithm: %s\n", coreset_algorithm.c_str());
            return 1;
        }


    } else if (strcmp(project_name, "RunKMeans") == 0) {

        if (argc < 12) {
            printf("Invalid number of arguments.\n");
            printf("Usage: dataset dataset_path is_coreset k max_iter init_kmeanspp tol n_init n_generation random_seed output_dir\n");
            std::cout << "\t- dataset =                 dataset name" << std::endl;
            std::cout << "\t- dataset_path =            path to the dataset (csv file)" << std::endl;
            std::cout << "\t- is_coreset =              1 if the input points are a coreset (start with weight, "
                         "d+1 columns), 0 otherwise" << std::endl;
            std::cout << "\t- k =                       number of clusters" << std::endl;
            std::cout << "\t- max_iter =                maximum number of iterations" << std::endl;
            std::cout << "\t- init_kmeanspp =           use kmeans++ initialization" << std::endl;
            std::cout << "\t- tol =                     tolerance" << std::endl;
            std::cout << "\t- n_init =                  number of initializations" << std::endl;
            std::cout << "\t- n_generation =            number of times for which running and writing outputs"
                      << std::endl;
            std::cout << "\t- random_seed =             random seed for reproducibility" << std::endl;
            std::cout << "\t- output_path =             output path" << std::endl;
            return 1;
        }

        const char *dataset_name(argv[1]);
        std::string dataset_path(argv[2]);
        int is_coreset = std::stoi(argv[3]);
        int k = std::stoi(argv[4]);
        int max_iter = std::stoi(argv[5]);
        bool init_kmeanspp = std::stoi(argv[6]);
        double tol = std::stod(argv[7]);
        int n_init = std::stoi(argv[8]);
        int n_generation = std::stoi(argv[9]);
        size_t random_seed = std::stol(argv[10]);
        std::string output_path(argv[11]);

        std::shared_ptr<DataParser> dataParser;
        if (match_dataset(dataset_name))
            dataParser = std::make_shared<CsvParser>();
        else {
            printf("Unknown dataset: %s\n", dataset_name);
            return 1;
        }

        // -- seed code
        Random::initialize(random_seed);

        std::shared_ptr<blaze::DynamicMatrix < double>>
        data;

        double time_to_parse_data = 0.0;
        if (is_coreset == 1) {
            // -- coreset: remove the first column of data
            time_to_parse_data = parse_data(dataset_path, dataParser, data, 1);
        } else {
            time_to_parse_data = parse_data(dataset_path, dataParser, data);
        }

        // -- run kmeans
        for (int idx_gen = 0; idx_gen < n_generation; idx_gen++) {
            printf("#%d generation (out of %d): ", idx_gen + 1, n_generation);
            StopWatch sw(true);
            KMeans kmeans(k, max_iter, init_kmeanspp, tol, n_init);
            auto clustering = kmeans.fit(*data, false);
            auto elapsed = sw.elapsedStr();
            auto elapsed_double = sw.elapsedSeconds();

            int iter_to_convergence = kmeans.get_iterations_to_convergence();
            double cost_kmeans = kmeans.get_best_cost();
            auto centers = clustering->get_centers();
            std::cout << "--> KMeans elapsed time: " << elapsed << std::endl;
            std::cout << "--> KMeans cost: " << cost_kmeans << std::endl;

            // -- write results
            write_kmeans_results(std::make_shared<blaze::DynamicMatrix < double>>
            (centers), cost_kmeans,
                    iter_to_convergence, elapsed_double, time_to_parse_data, is_coreset, idx_gen + 1,
                    output_path);

        }


    } else if (strcmp(project_name, "ComputeKMeansCost") == 0) {

        if (argc < 6) {
            printf("Invalid number of arguments.\n");
            printf("Usage: dataset input_points_path is_coreset list_of_centers_paths "
                   "(comma-separated csv files) output_path\n");
            std::cout << "\t- dataset =                 dataset name" << std::endl;
            std::cout << "\t- input_points_path =       path to the input points "
                         "(comma-separated csv file)" << std::endl;
            std::cout << "\t- is_coreset =              1 if the input points are a coreset (start with weight, "
                         "d+1 columns), 0 otherwise" << std::endl;
            std::cout << "\t- list_of_centers_path =    List of comma-separated paths to the centers "
                         "(comma-separated csv file)" << std::endl;
            std::cout << "\t- output_path =             output path (W = compute and save cost in center_path folder)"
                      << std::endl;
            return 1;
        }

        const char *dataset_name(argv[1]);
        std::string input_points_path(argv[2]);
        int is_coreset = std::stoi(argv[3]);
        // -- parse data (input points / coreset)
        std::shared_ptr<DataParser> dataParser;
        std::shared_ptr<blaze::DynamicMatrix < double>>
        data;

        if (is_coreset or match_dataset(dataset_name))
            dataParser = std::make_shared<CsvParser>();
        else {
            printf("Unknown dataset: %s\n", dataset_name);
            return 1;
        }

        // -- parse input (dataset or coreset)
        parse_data(input_points_path, dataParser, data);

        // -- parse list of centers
        std::string list_of_centers_path(argv[4]);
        std::string output_path(argv[5]);
        // -- if output_path = W -> activate flag for compute and save
        bool compute_and_save = false;
        if (output_path == "W") {
            compute_and_save = true;
        }
        std::vector<std::string> centers_paths;
        boost::split(centers_paths, list_of_centers_path, boost::is_any_of(","));
        // -- print size
        printf("Number of centers for which computing the cost: %lu\n", centers_paths.size());

        // -- assert existence of all centers paths
        for (const auto &center_path: centers_paths) {
            if (!boost::filesystem::exists(center_path)) {
                printf("Center path does not exist: %s\n", center_path.c_str());
                return 1;
            }
        }

        int idx_center = 0;
        for (const auto &center_path: centers_paths) {

            ++idx_center;
            std::cout << "Processing center #" << idx_center << " out of " << centers_paths.size() << std::endl;

            // -- parse centers
            std::shared_ptr<DataParser> centersParser = std::make_shared<CsvParser>();
            std::shared_ptr<blaze::DynamicMatrix < double>>
            centers;
            parse_data(center_path, centersParser, centers);
            std::string output_path_cost;
            double time_to_compute;

            StopWatch sw(true);

            // -- get the stem of center_path: apply boost stem twice
            std::string first_center_path_stem = boost::filesystem::path(center_path).stem().string();
            std::string center_path_stem = boost::filesystem::path(first_center_path_stem).stem().string();
            // -- get name of center_path, i.e., split by '.' and take the first element
            std::vector<std::string> center_path_split;
            boost::split(center_path_split, center_path, boost::is_any_of("."));
            std::string center_path_name = center_path_split[0];

            double cost;

            if (is_coreset == 1) {
                // -- coreset
                std::shared_ptr<Coreset> coreset = Coreset::create_from_data(data);
                cost = coreset->compute_kmeans_cost(*centers);
                time_to_compute = sw.elapsedSeconds();
                std::cout << "KMeans cost (coreset): " << cost << " in " << time_to_compute << " seconds" << std::endl;
                if (compute_and_save)
                    output_path_cost = center_path_name + "_coreset_cost.txt";
                else
                    output_path_cost = output_path + "/" + center_path_stem + "_coreset_cost.txt";

            } else {

                auto n = data->rows();
                auto d = data->columns();
                int k = (int) centers->rows();

                assert(d == centers->columns());

                // -- compute cost through assignment
                ClusteringAssignment assignment(n, k);
                assignment.assign_all(*data, *centers);
                cost = assignment.get_kmeans_cost();
                time_to_compute = sw.elapsedSeconds();
                std::cout << "KMeans cost: " << cost << " in " << time_to_compute << " seconds" << std::endl;
                //                std::stringstream ss;
                //                ss << std::setw(6) << std::setfill('0') << idx_center;
                if (compute_and_save) {
                    output_path_cost = center_path_name + "_kmeans_cost.txt";
                    // printf("Writing cost to %s\n", output_path_cost.c_str());
                } else
                    output_path_cost = output_path + "/" + center_path_stem + "_kmeans_cost.txt";

            }

            // -- write cost
            std::ofstream out_cost(output_path_cost);
            out_cost << std::fixed << cost << "," << time_to_compute;
            out_cost.close();


        }


    } else if (strcmp(project_name, "CandidatesGenerator") == 0) {

        if (argc < 11) {
            printf("Invalid number of arguments.\n");
            printf("Usage: dataset dataset_path is_coreset k n_generation_kmpp n_generation_random n_generation_ch "
                   "n_generation_meb random_seed output_path\n");
            std::cout << "\t- dataset =                 dataset name" << std::endl;
            std::cout << "\t- dataset_path =            path to the dataset (csv file)" << std::endl;
            std::cout << "\t- is_coreset =              1 if the input points are a coreset (start with weight, "
                         "d+1 columns), 0 otherwise" << std::endl;
            std::cout << "\t- k =                       number of centers to generate" << std::endl;
            std::cout << "\t- n_generation_kmpp =       number of times for which computing KMeans++ generation"
                      << std::endl;
            std::cout << "\t- n_generation_random =     number of times for which computing Random generation"
                      << std::endl;
            std::cout << "\t- n_generation_ch =         number of times for which computing ConvexHull generation"
                      << std::endl;
            std::cout
                    << "\t- n_generation_meb =        number of times for which computing Minimum Enclosing Ball (MEB) generation"
                    << std::endl;
            std::cout << "\t- random_seed =             random seed for reproducibility" << std::endl;
            std::cout << "\t- output_path =             output path" << std::endl;
            return 1;
        }

        const char *dataset_name(argv[1]);
        std::string dataset_path(argv[2]);
        const int is_coreset = std::stoi(argv[3]);
        const int k = std::stoi(argv[4]);
        const int n_generation_kmpp = std::stoi(argv[5]);
        const int n_generation_random = std::stoi(argv[6]);
        const int n_generation_ch = std::stoi(argv[7]);
        const int n_generation_meb = std::stoi(argv[8]);
        const size_t random_seed = std::stol(argv[9]);
        std::string output_dir(argv[10]);

        printf("Running with the following parameters:\n");
        printf(" - Dataset: %s\n", dataset_name);
        printf(" - Dataset path: %s\n", dataset_path.c_str());
        printf(" - Is coreset: %d\n", is_coreset);
        printf(" - Number of centers to generate: %d\n", k);
        printf(" - Number of times for which computing KMeans++ generation: %d\n", n_generation_kmpp);
        printf(" - Number of times for which computing Random generation: %d\n", n_generation_random);
        printf(" - Number of times for which computing ConvexHull generation: %d\n", n_generation_ch);
        printf(" - Number of times for which computing Minimum Enclosing Ball (MEB) generation: %d\n",
               n_generation_meb);
        printf(" - Output directory: %s\n", output_dir.c_str());

        // -- parse data
        std::shared_ptr<DataParser> dataParser;
        if (match_dataset(dataset_name))
            dataParser = std::make_shared<CsvParser>();
        else {
            printf("Unknown dataset: %s\n", dataset_name);
            return 1;
        }

        std::shared_ptr<blaze::DynamicMatrix < double>>
        data;
        parse_data(dataset_path, dataParser, data, is_coreset);

        const int total_generations = n_generation_kmpp + n_generation_random + n_generation_ch + n_generation_meb;
        printf("Total generations: %d\n", total_generations);

        std::vector<std::pair<std::string, centers>> centers_list;
        int current_generation = 0;

        // -- generate candidates kmpp
        for (int idx_gen = 0; idx_gen < n_generation_kmpp; idx_gen++) {
            printf("---------- KMeans++ Generation %d ----------\n", idx_gen + 1);
            centers_list.emplace_back("kmpp", generate_candidates("kmpp", *data, k, random_seed + current_generation));
            current_generation++;
        }

        // -- generate candidates random
        for (int idx_gen = 0; idx_gen < n_generation_random; idx_gen++) {
            printf("---------- Random Generation %d ----------\n", idx_gen + 1);
            centers_list.emplace_back("random",
                                      generate_candidates("random", *data, k, random_seed + current_generation));
            current_generation++;
        }

        // -- generate candidates ch
        for (int idx_gen = 0; idx_gen < n_generation_ch; idx_gen++) {
            printf("---------- Convex Hull (CH) Generation %d ----------\n", idx_gen + 1);
            centers_list.emplace_back("ch", generate_candidates("ch", *data,
                                                                k, random_seed + current_generation));
            current_generation++;
        }

        // -- generate candidates meb
        for (int idx_gen = 0; idx_gen < n_generation_meb; idx_gen++) {
            printf("---------- Minimum Enclosing Ball (MEB) Generation %d ----------\n", idx_gen + 1);
            centers_list.emplace_back("meb", generate_candidates("meb", *data,
                                                                 k, random_seed + current_generation));
            current_generation++;
        }


        // -- write centers
        for (int idx_gen = 0; idx_gen < total_generations; idx_gen++) {
            std::stringstream ss;
            ss << std::setw(3) << std::setfill('0') << idx_gen + 1;
            auto elem = centers_list[idx_gen];
            auto output_path_centers = output_dir + "/" + ss.str() + "_" + elem.first + "_candidate_centers" + ".txt";
            write_assignment_centers(elem.second, output_path_centers);
        }

    } else {
        printf("Unknown project: %s\n", project_name);
        printf("Available projects: CoresetWithPredictions, RunKMeans, ComputeKMeansCost, CoresetGenerator\n");
        return 1;
    }
    return 0;
}
