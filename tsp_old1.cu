#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/find.h>
#include <fstream>
#include <sstream>
#include <string>

struct City {
    int id;
    double x, y;
};

std::vector<City> readTSPFile(const std::string& filename) {
    std::vector<City> cities;
    std::ifstream file(filename);
    std::string line;
    bool inNodeSection = false;

    if (file.is_open()) {
        while (getline(file, line)) {
            if (line.find("NODE_COORD_SECTION") != std::string::npos) {
                inNodeSection = true;
                continue;
            }
            if (line.find("EOF") != std::string::npos) {
                break;
            }
            if (inNodeSection) {
                std::istringstream iss(line);
                City city;
                iss >> city.id >> city.x >> city.y;
                cities.push_back(city);
            }
        }
        file.close();
    } else {
        std::cerr << "Unable to open file" << std::endl;
    }
    return cities;
}

double host_distance(const City& a, const City& b) {
    return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

double host_pathDistance(const std::vector<City>& cities, const std::vector<int>& path) {
    double totalDistance = 0.0;
    for (size_t i = 0; i < path.size() - 1; ++i) {
        totalDistance += host_distance(cities[path[i]], cities[path[i + 1]]);
    }
    totalDistance += host_distance(cities[path.back()], cities[path[0]]);
    return totalDistance;
}

double host_fitness(const std::vector<City>& cities, const std::vector<int>& path) {
    return 1.0 / host_pathDistance(cities, path);
}

__device__ double distance(const City& a, const City& b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

__device__ double pathDistance(const City* cities, const int* path, int numCities) {
    double totalDistance = 0.0;

    for (int i = 0; i < numCities - 1; ++i) {
        totalDistance += distance(cities[path[i]], cities[path[i + 1]]);
    }
    totalDistance += distance(cities[path[numCities - 1]], cities[path[0]]);
    return totalDistance;
}

__global__ void calculateFitness(const City* cities, const int* population, double* fitnessValues, int numCities, int populationSize) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < populationSize; i += stride) {
        fitnessValues[idx] = 1.0 / pathDistance(cities, &population[idx * numCities], numCities);
    }
}

std::vector<std::vector<int>> initializePopulation(int populationSize, int numCities) {
    std::vector<std::vector<int>> population(populationSize, std::vector<int>(numCities));
    std::vector<int> basePath(numCities);
    for (int i = 0; i < numCities; ++i) {
        basePath[i] = i;
    }
    for (auto& individual : population) {
        std::shuffle(basePath.begin(), basePath.end(), std::mt19937{std::random_device{}()});
        individual = basePath;
    }
    return population;
}

std::vector<int> flattenPopulation(const std::vector<std::vector<int>>& population) {
    std::vector<int> flatPopulation;
    for (const auto& individual : population) {
        flatPopulation.insert(flatPopulation.end(), individual.begin(), individual.end());
    }
    return flatPopulation;
}

std::vector<std::vector<int>> unflattenPopulation(const int* flatPopulation, int populationSize, int numCities) {
    std::vector<std::vector<int>> population(populationSize, std::vector<int>(numCities));
    for (int i = 0; i < populationSize; ++i) {
        for (int j = 0; j < numCities; ++j) {
            population[i][j] = flatPopulation[i * numCities + j];
        }
    }
    return population;
}

__device__ void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}


__device__ int tournamentSelection(const double* fitnessValues, int populationSize, curandState* state) {
    int tournamentSize = 5;
    int best = -1;
    double bestFitness = -1.0;
    for (int i = 0; i < tournamentSize; ++i) {
        int idx = curand(state) % populationSize;
        if (fitnessValues[idx] > bestFitness) {
            bestFitness = fitnessValues[idx];
            best = idx;
        }
    }
    return best;
}


__device__ void orderCrossover(const int* parent1, const int* parent2, int* child, int numCities, curandState* state) {
    int size = numCities;
    int start = curand(state) % size;
    int end = curand(state) % size;

    if (start > end) swap(start, end);

    for (int i = 0; i < size; ++i) {
        child[i] = -1;
    }

    for (int i = start; i <= end; ++i) {
        child[i] = parent1[i];
    }

    int current = (end + 1) % size;
    for (int i = 0; i < size; ++i) {
        int idx = (end + 1 + i) % size;
        if (thrust::find(thrust::seq, child, child + size, parent2[idx]) == child + size) {
            child[current] = parent2[idx];
            current = (current + 1) % size;
        }
    }
}


__device__ void mutate(int* individual, int numCities, curandState* state) {
    int size = numCities;
    int a = curand(state) % size;
    int b = curand(state) % size;
    swap(individual[a], individual[b]);
}


__global__ void setupCurandStates(curandState* states, unsigned long seed, int populationSize) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < populationSize; i += stride) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}


__global__ void geneticAlgorithmKernel(const City* cities, int* population,  int* next_population, double* fitnessValues, int numCities, int populationSize, curandState* states) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < populationSize; i += stride)  {
        curandState* state = &states[idx];

        int parent1Idx = tournamentSelection(fitnessValues, populationSize, state);
        int parent2Idx = tournamentSelection(fitnessValues, populationSize, state);

        const int* parent1 = &population[parent1Idx * numCities];
        const int* parent2 = &population[parent2Idx * numCities];
        int* child = &next_population[idx * numCities];

        orderCrossover(parent1, parent2, child, numCities, state);

        if (curand(state) % 100 < 10) {
            mutate(child, numCities, state);
        }
    }
}

__global__ void applyNextGen(const City* cities, int* population,  int* next_population, double* fitnessValues, int numCities, int populationSize, curandState* states) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < populationSize; i += stride)  {
        int* ori = &population[i * numCities];
        int* child = &next_population[i * numCities];

        for (int j = 0; j < numCities; ++j) {
            ori[j] = child[j];
        }
    }
}


std::vector<int> geneticAlgorithm(const std::vector<City>& cities, int populationSize, int generations) {
    int numCities = cities.size();

    std::vector<std::vector<int>> population = initializePopulation(populationSize, numCities);
    std::vector<int> flatPopulation = flattenPopulation(population);
    std::vector<double> fitnessValues(populationSize);

    City* d_cities;
    int* d_population, *d_next_population;
    double* d_fitnessValues;
    curandState* d_states;

    cudaMalloc((void**)&d_cities, numCities * sizeof(City));
    cudaMalloc((void**)&d_population, populationSize * numCities * sizeof(int));
    cudaMalloc((void**)&d_next_population, populationSize * numCities * sizeof(int));
    cudaMalloc((void**)&d_fitnessValues, populationSize * sizeof(double));
    cudaMalloc((void**)&d_states, populationSize * sizeof(curandState));

    cudaMemcpy(d_cities, cities.data(), numCities * sizeof(City), cudaMemcpyHostToDevice);
    cudaMemcpy(d_population, flatPopulation.data(), populationSize * numCities * sizeof(int), cudaMemcpyHostToDevice);

    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    size_t blockSize = 512;
    size_t numBlocks = 32 * numberOfSMs;

    setupCurandStates<<<numBlocks, blockSize>>>(d_states, time(0), populationSize);
    cudaDeviceSynchronize();

    for (int gen = 0; gen < generations; ++gen) {
        
        std::cout << "Gen: " << gen << std::endl;

        calculateFitness<<<numBlocks, blockSize>>>(d_cities, d_population, d_fitnessValues, numCities, populationSize);
        cudaDeviceSynchronize();

        geneticAlgorithmKernel<<<numBlocks, blockSize>>>(d_cities, d_population, d_next_population, d_fitnessValues, numCities, populationSize, d_states);
        cudaDeviceSynchronize();

        applyNextGen<<<numBlocks, blockSize>>>(d_cities, d_population, d_next_population, d_fitnessValues, numCities, populationSize, d_states);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(flatPopulation.data(), d_population, populationSize * numCities * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_cities);
    cudaFree(d_population);
    cudaFree(d_fitnessValues);
    cudaFree(d_states);

    population = unflattenPopulation(flatPopulation.data(), populationSize, numCities);

    int bestIndex = 0;
    double bestFitness = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < populationSize; ++i) {
        double fit = host_fitness(cities, population[i]);
        if (fit > bestFitness) {
            bestFitness = fit;
            bestIndex = i;
        }
    }
    return population[bestIndex];
}

int main() {

    // std::vector<City> cities = {
    //     {0, 60, 200}, {1, 180, 200}, {2, 80, 180}, {3, 140, 180}, {4, 20, 160}, 
    //     {5, 100, 160}, {6, 200, 160}, {7, 140, 140}, {8, 40, 120}, {9, 100, 120},
    //     {10, 180, 100}, {11, 60, 80}, {12, 120, 80}, {13, 180, 60}, {14, 20, 40},
    //     {15, 100, 40}, {16, 200, 40}, {17, 20, 20}, {18, 60, 20}, {19, 160, 20}
    // };

    std::string filename = "assets/qa194.tsp";
    std::vector<City> cities = readTSPFile(filename);

    int populationSize = 1000;
    int generations = 1000;

    std::vector<int> bestPath = geneticAlgorithm(cities, populationSize, generations);

    std::cout << "Best path: ";
    for (int city : bestPath) {
        std::cout << city << " ";
    }
    std::cout << std::endl;

    std::cout << "Total distance: " << host_pathDistance(cities, bestPath) << std::endl;

    return 0;
}
