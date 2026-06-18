cmake_minimum_required(VERSION 3.5)

project(
        CoresetWithPredictions
        VERSION 1.0
        LANGUAGES CXX
)

message(STATUS "Started CMake for ${PROJECT_NAME}...\n")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(KMEANS_EXECUTABLE RunKMeans)
set(KMEANS_COST_EXECUTABLE ComputeKMeansCost)
set(CANDIDATES_GENERATOR CandidatesGenerator)

include(cmake/SetSourcesHeader.cmake)

# -- Targets
add_executable(${PROJECT_NAME} ${headers} ${sources})
add_executable(${KMEANS_EXECUTABLE} ${headers} ${sources})
add_executable(${KMEANS_COST_EXECUTABLE} ${headers} ${sources})
add_executable(${CANDIDATES_GENERATOR} ${headers} ${sources})

set(ALL_TARGETS ${PROJECT_NAME} ${KMEANS_EXECUTABLE} ${KMEANS_COST_EXECUTABLE}  ${CANDIDATES_GENERATOR})

# -- Include directories
foreach(TARGET IN LISTS ALL_TARGETS)
    target_include_directories(${TARGET}
            PUBLIC
            $<INSTALL_INTERFACE:include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/source
    )
endforeach()

target_compile_definitions(${PROJECT_NAME} PRIVATE MAIN_PROJECT_BUILD)

# -- Blaze
# Pass -DBLAZE_DIR=<prefix> to cmake if Blaze is not in a standard location.
find_package(blaze REQUIRED)
message(STATUS "Blaze found!")

foreach(TARGET IN LISTS ALL_TARGETS)
    target_link_libraries(${TARGET} PRIVATE blaze::blaze)
endforeach()

# -- Boost
# Pass -DBOOST_ROOT=<prefix> to cmake if Boost is not in a standard location.
find_package(Boost REQUIRED COMPONENTS iostreams filesystem)
message(STATUS "Boost libraries found!")

foreach(TARGET IN LISTS ALL_TARGETS)
    target_link_libraries(${TARGET} PRIVATE
            Boost::iostreams
            Boost::filesystem
    )
endforeach()

# -- OpenMP
find_package(OpenMP REQUIRED)
message(STATUS "OpenMP found!")

foreach(TARGET IN LISTS ALL_TARGETS)
    target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
endforeach()

# -- Compiler options (e.g., O3 optimization)
foreach(TARGET IN LISTS ALL_TARGETS)
    target_compile_options(${TARGET} PRIVATE
            -fexceptions
            $<$<CONFIG:Debug>:-O3;-g3;-ggdb>
    )
endforeach()