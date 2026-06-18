//
// Created by X on 03/10/24.
//

#pragma once

#include <iostream>
#include <sstream>
#include <fstream>
#include <ctime>
#include <chrono>
#include <iomanip>

using namespace std::chrono;

class StopWatch {
private:
    std::chrono::high_resolution_clock::time_point startTime;

public:
    StopWatch(bool startWatch = false);

    void start();

    std::string elapsedStr();

    double elapsedSeconds();
};


