# Sensitivity Sampling with Predictions for $k$-Means Clustering (ECML-PKDD 2026)
Repository containing code for *"Sensitivity Sampling with Predictions for* $k$*-Means Clustering"*, accepted at ECML-PKDD Conference (2026).

Overall, the project provides the following functions, used in the experiments section of the paper:
- Building coresets from input datasets (`CoresetWithPredictions` executable),
- Running $k$-means (LLoyd's algorithm) on full datasets or coresets (`RunKMeans` executable),
- Computing the $k$-means cost of given centers (`ComputeKMeansCost` executable),
- Generating candidate centers for assessing distortion of coresets (`CandidatesGenerator` executable).

---

## Datasets

Inside `data` folder, we provide Python script for creating snapshots from our considered datasets (i.e., discretizing time information using custom intervals such us days, months or years). 
In the following, we provide a table reporting statistics and download links of our considered datasets.

| Dataset | Total Points | # Snapshots | Aggregation | $n_{max}$ | $d$ |
| --- | ---: | ---: | --- | ---: | ---: |
| [Twitter](https://archive.ics.uci.edu/dataset/1050/twitter+geospatial+data) | 14M | 7 | Daily | 2.2M | 3 |
| [IntelLab](https://db.csail.mit.edu/labdata/labdata.html) | 2.2M | 34 | Daily | 101k | 6 |
| [Taxi](https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015) | 1.7M | 12 | Monthly | 161k | 2 |
| [NYC TLC (M)](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data) | 47M | 4 | Monthly | 12.7M | 16 |
| [NYC TLC (Y)](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | 743M | 10 | Yearly | 146M | 14 |

---

## Code Requirements

The code is written in C++17. The project relies on: GCC / G++, CMake, Boost, Blaze, BLAS / LAPACK, OpenMP, zlib. 
Our code is inspired by: [Schwiegelshohn, Chris, and Omar Ali Sheikh-Omar. "An Empirical Evaluation of $ k $-Means Coresets." ESA (2022)](https://github.com/sheikhomar/eval-k-means-coresets). 

The repository includes an installation script for Ubuntu/Debian-like systems.

## Install prerequisites

Before compiling the project, run:

``bash install_prerequisites.sh``

This script installs the required system dependencies and then installs: CMake 3.20.2, Boost 1.89.0, Blaze.


## Compilation

After the installation of prerequisites, run the following command from the `code` folder:

``bash compile.sh``

The binaries will be generated automatically inside the ``build/`` directory.


### Supported datasets

All datasets are currently parsed through CsvParser, which handles .csv files. The accepted dataset names are matched by prefix.

### Available coreset algorithms

The following coreset construction algorithms are supported:
- uniform-sampling
- sensitivity-sampling, accepting the following flags
  - *R*: Read the centers from file (behaviour of our algorithm *PreSenS*) for deriving the bi-criteria approximation
  - *C* (or *W*): Computes from scratch the bi-criteria approximation, and Writes to file if W enabled (behaviour of *SenS* algorithm and *S-20kmpp* when initialized with n_init = 20, see below)

---

## 1. Build a coreset

Executable: `./build/CoresetWithPredictions`

### 1.1 Uniform sampling

Usage:

``./build/CoresetWithPredictions <dataset> <dataset_path> uniform-sampling <m> <k> <random_seed> <output_dir>``

Parameters:

- dataset: dataset name or valid prefix
- dataset_path: path to the input dataset
- m: coreset size
- k: number of clusters
- random_seed: random seed for reproducibility
- output_dir: directory where the output files will be saved

Output:

- coreset_data.txt.gz
- times.txt


### 1.2 Sensitivity sampling

Usage:

``./build/CoresetWithPredictions <dataset> <dataset_path> sensitivity-sampling <m> <k> <random_seed> <output_dir> <predictor_flag> <predictor_path> <n_times_kmpp>``

Parameters:

- dataset: dataset name or valid prefix
- dataset_path: path to the input dataset
- m: coreset size
- k: number of clusters
- random_seed: random seed for reproducibility
- output_dir: directory where the output files will be saved
- predictor_flag: mode for handling predictor centers
  - R: read predictor centers from file
  - C: compute predictor centers without writing them to file
  - W: compute predictor centers and write them to file when n_times_kmpp = 1
- predictor_path: path to the predictor file
- n_times_kmpp: number of oracle assignments to compute when oracle_flag is C or W

Behavior:

- If oracle_flag = R, the program reads the predictor centers from predictor_path.
- Otherwise, it computes n_times_kmpp assignments using 2k centers.
- If oracle_flag = W and n_times_kmpp = 1, the computed predictor centers are written to predictor_path.

Output:

- coreset_data.txt.gz
- times.txt


## 2. Run k-means

Executable: `./build/RunKMeans`

Usage:

``./build/RunKMeans <dataset> <dataset_path> <is_coreset> <k> <max_iter> <init_kmeanspp> <tol> <n_init> <n_generation> <random_seed> <output_dir>``

Parameters:

- dataset: dataset name or valid prefix
- dataset_path: path to the input dataset or coreset
- is_coreset: 1 if the input points are a coreset, 0 otherwise
- k: number of clusters
- max_iter: maximum number of k-means iterations
- init_kmeanspp: 1 to use k-means++ initialization, 0 otherwise
- tol: convergence tolerance
- n_init: number of initializations
- n_generation: number of independent runs to execute and save
- random_seed: random seed for reproducibility
- output_dir: directory where the output files will be saved

## 3. Compute k-means cost

Executable: `./build/ComputeKMeansCost`

Usage:

``./build/ComputeKMeansCost <dataset> <input_points_path> <is_coreset> <list_of_centers_paths> <output_path>``

Parameters:

- dataset: dataset name or valid prefix
- input_points_path: path to the input dataset or coreset
- is_coreset: 1 if the input file is a coreset, 0 otherwise
- list_of_centers_paths: comma-separated list of paths to centers files
- output_path:
  - either a directory where the cost files will be written,
  - or W, in which case the program writes each cost file next to the corresponding centers file prefix


## 4. Generate candidate centers

Executable: `./build/CandidatesGenerator`

Usage:

``./build/CandidatesGenerator <dataset> <dataset_path> <is_coreset> <k> <n_generation_kmpp> <n_generation_random> <n_generation_ch> <n_generation_meb> <random_seed> <output_dir>``

Parameters:

- dataset: dataset name or valid prefix
- dataset_path: path to the input dataset or coreset
- is_coreset: 1 if the input file is a coreset, 0 otherwise
- k: number of centers to generate
- n_generation_kmpp: number of candidate sets generated with k-means++
- n_generation_random: number of candidate sets generated uniformly at random
- n_generation_ch: number of candidate sets generated with the convex-hull strategy
- n_generation_meb: number of candidate sets generated with the minimum-enclosing-ball strategy
- random_seed: random seed for reproducibility
- output_dir: directory where the candidate centers will be saved

Supported candidate generators: kmpp, random, ch, meb.

