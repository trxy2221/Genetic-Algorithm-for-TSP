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
#include <chrono>
#include <omp.h>

struct City {
    double x, y;
};

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

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
                int tmp;
                iss >> tmp >> city.x >> city.y;
                cities.push_back(city);
            }
        }
        file.close();
    } else {
        std::cerr << "Unable to open file" << std::endl;
    }
    return cities;
}


void showProgress(int current, int total, std::chrono::steady_clock::time_point startTime) {
    int width = 10;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(width * progress);

    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = now - startTime;
    double estimatedTotalTime = elapsed.count() / progress;
    double remainingTime = estimatedTotalTime - elapsed.count();

    printf("[");
    for (int i = 0; i < width; ++i) {
        if (i < pos) printf("=");
        else if (i == pos) printf(">");
        else printf(" ");
    }
    printf("] %.1f%% Elapsed: %.1fs Remaining: %.1fs\r", progress * 100.0, elapsed.count(), remainingTime);
    fflush(stdout);
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

__global__ void calculateDistances(const City* cities, double* distanceMatrix, int numCities) {
    
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < numCities * numCities; i += stride) {
        int x = i / numCities;
        int y = i % numCities;
        double dx = cities[x].x - cities[y].x;
        double dy = cities[x].y - cities[y].y;
        distanceMatrix[i] = std::sqrt(dx * dx + dy * dy);
        // printf("(%d,%d) = %f\n", x, y, distanceMatrix[i]);
    }
}

__device__ double pathDistance(const City* cities, const double* distanceMatrix, const int* path, int numCities) {
    
    double totalDistance = 0.0;
    for (int i = 0; i < numCities - 1; ++i) {
        totalDistance += distanceMatrix[path[i] * numCities + path[i + 1]];
    }
    totalDistance += distanceMatrix[path[0] * numCities + path[numCities - 1]];
    return totalDistance;
}

__global__ void calculateFitness(const City* cities, double* distanceMatrix, const int* population, double* fitnessValues, int numCities, int populationSize) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < populationSize; i += stride) {
        fitnessValues[idx] = 1.0 / pathDistance(cities, distanceMatrix, &population[idx * numCities], numCities);
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


__device__ int tournamentSelection(const double* fitnessValues, int populationSize, unsigned int childId, int islandSize, curandState* state) {
    int tournamentSize = 5;
    int best = -1;
    double bestFitness = -1.0;
    int childBlock = childId / islandSize;

    for (int i = 0; i < tournamentSize; ++i) {
        int idx = childBlock * islandSize + curand(state) % islandSize ;
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
    int start = curand(state) % size;
    int end = curand(state) % size;
    int a, b;
    for(int i = 0; i <= (end - start) / 2; i++){
        a = start + i;
        b = end - i;
        if(a != b){
            swap(individual[a], individual[b]);
        }
    } 
}


__global__ void setupCurandStates(curandState* states, unsigned long seed, int populationSize) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < populationSize; i += stride) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}


__global__ void geneticAlgorithmKernel(const City* cities, int* population, int* next_population, double* distanceMatrix, double* fitnessValues, double* next_fitnessValues, int numCities, int populationSize, int islandSize, curandState* states) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < populationSize; i += stride)  {
        curandState* state = &states[idx];

        int parent1Idx = tournamentSelection(fitnessValues, populationSize, i, islandSize, state);
        int parent2Idx = tournamentSelection(fitnessValues, populationSize, i, islandSize, state);
        long p1 = parent1Idx * numCities;
        long p2 = parent2Idx * numCities;

        const int* parent1 = &population[p1];
        const int* parent2 = &population[p2];
        int* child = &next_population[i * numCities];

        orderCrossover(parent1, parent2, child, numCities, state);

        if (curand(state) % 100 < 10) {
            mutate(child, numCities, state);
        }

        next_fitnessValues[idx] = 1.0 / pathDistance(cities, distanceMatrix, child, numCities);
    }
}

__global__ void applyNextGen(const City* cities, int* population,  int* next_population, double* fitnessValues, double* next_fitnessValues, int numCities, int populationSize, curandState* states) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < populationSize; i += stride)  {
        int* ori = &population[i * numCities];
        int* child = &next_population[i * numCities];
        if(next_fitnessValues[i] >= fitnessValues[i]){
            for (int j = 0; j < numCities; ++j) {
                ori[j] = child[j];
            }
            fitnessValues[i] = next_fitnessValues[i];
        }
    }
}


std::vector<int> geneticAlgorithm(const std::vector<City>& cities, int populationSize, int generations) {
    
    int numCities = cities.size();

    //define population1 and population2 in CPU
    std::vector<std::vector<int>> population1, population2;
    std::vector<int> flatPopulation1, flatPopulation2;
    
    //define pointers for GPU 0
    City* d_cities1;
    int* d_population1, *d_next_population1;
    double* d_fitnessValues1, *d_next_fitnessValues1;
    double* d_distanceMatrix1;
    curandState* d_states1;

    //define pointers for GPU 1
    City* d_cities2;
    int* d_population2, *d_next_population2;
    double* d_fitnessValues2, *d_next_fitnessValues2;
    double* d_distanceMatrix2;
    curandState* d_states2;

    //variables for counting block and threads
    int deviceId, numberOfSMs;
    size_t blockSize, numBlocks, numThreads;

    //variable for selecting best path
    double bestFitness1 = -std::numeric_limits<double>::infinity();
    double bestFitness2 = -std::numeric_limits<double>::infinity();

    std::vector<int> bestpath1;
    std::vector<int> bestpath2;

    int numIslands = 2;
    int islandSize = populationSize / numIslands;
    
    #pragma omp parallel num_threads(2)
    {
        int thread_id = omp_get_thread_num();

        if(thread_id == 0){

            // GPU 0
            cudaSetDevice(0);

            //initial population1 in CPU
            population1 = initializePopulation(populationSize, numCities);
            flatPopulation1 = flattenPopulation(population1);
 
            //malloc in GPU 0 
            cudaMalloc((void**)&d_cities1, numCities * sizeof(City));
            cudaMalloc((void**)&d_population1, populationSize * numCities * sizeof(int));
            cudaMalloc((void**)&d_next_population1, populationSize * numCities * sizeof(int));
            cudaMalloc((void**)&d_fitnessValues1, populationSize * sizeof(double));
            cudaMalloc((void**)&d_next_fitnessValues1, populationSize * sizeof(double));
            cudaMalloc((void**)&d_states1, populationSize * sizeof(curandState));
            cudaMalloc(&d_distanceMatrix1, numCities * numCities * sizeof(double));

            //memory copy dities and population to GPU 0
            cudaMemcpy(d_cities1, cities.data(), numCities * sizeof(City), cudaMemcpyHostToDevice);
            cudaMemcpy(d_population1, flatPopulation1.data(), populationSize * numCities * sizeof(int), cudaMemcpyHostToDevice);

            cudaGetDevice(&deviceId);
            if(deviceId == 0) printf("GPU 0 working!!!\n");
            cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

            blockSize = 512;
            numBlocks = 32 * numberOfSMs;
            numThreads = blockSize * numBlocks;

            std::cout << "block size      : " << blockSize << std::endl;
            std::cout << "num of blocks   : " << numBlocks << std::endl;
            std::cout << "total threads   : " << numThreads << std::endl;
            std::cout << "====================================\n" << std::endl;

            //set random state for GPU 0 
            setupCurandStates<<<numBlocks, blockSize>>>(d_states1, time(0), populationSize);
            // cudaDeviceSynchronize();

            calculateDistances<<<numBlocks, blockSize>>>(d_cities1, d_distanceMatrix1, numCities);
            cudaDeviceSynchronize();

        } else {

            // GPU 1
            cudaSetDevice(1);

            //initial population2 in CPU
            population2 = initializePopulation(populationSize, numCities);
            flatPopulation2 = flattenPopulation(population2);

            //malloc in GPU 1 
            cudaMalloc((void**)&d_cities2, numCities * sizeof(City));
            cudaMalloc((void**)&d_population2, populationSize * numCities * sizeof(int));
            cudaMalloc((void**)&d_next_population2, populationSize * numCities * sizeof(int));
            cudaMalloc((void**)&d_fitnessValues2, populationSize * sizeof(double));
            cudaMalloc((void**)&d_next_fitnessValues2, populationSize * sizeof(double));
            cudaMalloc((void**)&d_states2, populationSize * sizeof(curandState));
            cudaMalloc(&d_distanceMatrix2, numCities * numCities * sizeof(double));

            //memory copy dities and population to GPU 1
            cudaMemcpy(d_cities2, cities.data(), numCities * sizeof(City), cudaMemcpyHostToDevice);
            cudaMemcpy(d_population2, flatPopulation1.data(), populationSize * numCities * sizeof(int), cudaMemcpyHostToDevice);

            cudaGetDevice(&deviceId);
            if(deviceId == 1) printf("GPU 1 working!!!\n");
            cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

            blockSize = 512;
            numBlocks = 32 * numberOfSMs;

            //set random state for GPU 1 
            setupCurandStates<<<numBlocks, blockSize>>>(d_states2, time(0), populationSize);
            // cudaDeviceSynchronize();

            calculateDistances<<<numBlocks, blockSize>>>(d_cities2, d_distanceMatrix2, numCities);
            cudaDeviceSynchronize();
        }

        #pragma omp barrier

        //check is P2P avaible and enable P2P
        int canAccessPeer0, canAccessPeer1;
        if (thread_id == 0) {
            cudaDeviceCanAccessPeer(&canAccessPeer0, 0, 1);
            if (canAccessPeer0) {
                checkCudaError(cudaDeviceEnablePeerAccess(1, 0), "Failed to enable peer access from GPU 0 to GPU 1");
            }
        } else {
            cudaDeviceCanAccessPeer(&canAccessPeer1, 1, 0);
            if (canAccessPeer1) {
                checkCudaError(cudaDeviceEnablePeerAccess(0, 0), "Failed to enable peer access from GPU 1 to GPU 0");
            }
        }

        #pragma omp barrier

        auto startTime = std::chrono::steady_clock::now();

        for (int gen = 0; gen < generations; ++gen) {

            if(thread_id == 0){

                showProgress(gen, generations, startTime);

                //GPU 0
                calculateFitness<<<numBlocks, blockSize>>>(d_cities1, d_distanceMatrix1, d_population1, d_fitnessValues1, numCities, populationSize);
                cudaDeviceSynchronize();

                geneticAlgorithmKernel<<<numBlocks, blockSize>>>(d_cities1, d_population1, d_next_population1, d_distanceMatrix1, d_fitnessValues1, d_next_fitnessValues1, numCities, populationSize, islandSize, d_states1);
                cudaDeviceSynchronize();

                applyNextGen<<<numBlocks, blockSize>>>(d_cities1, d_population1, d_next_population1, d_fitnessValues1, d_next_fitnessValues1, numCities, populationSize, d_states1);
                cudaDeviceSynchronize();

            } else {

                //GPU 1
                calculateFitness<<<numBlocks, blockSize>>>(d_cities2, d_distanceMatrix2, d_population2, d_fitnessValues2, numCities, populationSize);
                cudaDeviceSynchronize();

                geneticAlgorithmKernel<<<numBlocks, blockSize>>>(d_cities2, d_population2, d_next_population2, d_distanceMatrix2, d_fitnessValues2, d_next_fitnessValues2, numCities, populationSize, islandSize, d_states2);
                cudaDeviceSynchronize();

                applyNextGen<<<numBlocks, blockSize>>>(d_cities2, d_population2, d_next_population2, d_fitnessValues2, d_next_fitnessValues2, numCities, populationSize, d_states2);
                cudaDeviceSynchronize();
            }

            // merge population1 and population 2 per 5000 generations
            if(gen % 1000 == 0 && gen != 0){

                #pragma omp barrier

                //pass self population to other side next_population
                if (thread_id == 0) {
                    printf("copy from GPU 0 to GPU 1 at gen: %d\n", gen);
                    if (canAccessPeer0) {
                        checkCudaError(cudaMemcpyPeer(d_next_population2, 1, d_population1, 0, populationSize * numCities * sizeof(int)), "Peer-to-peer memory copy from GPU 0 to GPU 1 failed");
                        checkCudaError(cudaMemcpyPeer(d_next_fitnessValues2, 1, d_fitnessValues1, 0, populationSize * sizeof(double)), "Peer-to-peer memory copy from GPU 0 to GPU 1 failed");
                    } else {
                        checkCudaError(cudaMemcpy(d_next_population2, d_population1, populationSize * numCities * sizeof(int), cudaMemcpyDeviceToDevice), "Fallback memory copy from GPU 0 to GPU 1 failed");
                        checkCudaError(cudaMemcpy(d_next_fitnessValues2, d_fitnessValues1, populationSize * sizeof(double), cudaMemcpyDeviceToDevice), "Fallback memory copy from GPU 0 to GPU 1 failed");
                    }

                } else {

                    printf("copy from GPU 1 to GPU 0 at gen: %d\n", gen);
                    if (canAccessPeer1) {
                        checkCudaError(cudaMemcpyPeer(d_next_population1, 0, d_population2, 1, populationSize * numCities * sizeof(int)), "Peer-to-peer memory copy from GPU 1 to GPU 0 failed");
                        checkCudaError(cudaMemcpyPeer(d_next_fitnessValues1, 0, d_fitnessValues2, 1, populationSize * sizeof(double)), "Peer-to-peer memory copy from GPU 1 to GPU 0 failed");
                    } else {
                        checkCudaError(cudaMemcpy(d_next_population1, d_population2, populationSize * numCities * sizeof(int), cudaMemcpyDeviceToDevice), "Fallback memory copy from GPU 1 to GPU 0 failed");
                        checkCudaError(cudaMemcpy(d_next_fitnessValues1, d_fitnessValues2, populationSize * sizeof(double), cudaMemcpyDeviceToDevice), "Fallback memory copy from GPU 1 to GPU 0 failed");
                    }
                }

                #pragma omp barrier

                //merge population and next population
                if(thread_id == 0){
                    applyNextGen<<<numBlocks, blockSize>>>(d_cities1, d_population1, d_next_population1, d_fitnessValues1, d_next_fitnessValues1, numCities, populationSize, d_states1);
                    cudaDeviceSynchronize();

                } else {
                    applyNextGen<<<numBlocks, blockSize>>>(d_cities2, d_population2, d_next_population2, d_fitnessValues2, d_next_fitnessValues2, numCities, populationSize, d_states2);
                    cudaDeviceSynchronize();
                } 
            }
        }

        // copy population back to CPU and select the best path
        if(thread_id == 0){

            showProgress(generations, generations, startTime);

            cudaMemcpy(flatPopulation1.data(), d_population1, populationSize * numCities * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(d_cities1);
            cudaFree(d_population1);
            cudaFree(d_next_population1);
            cudaFree(d_fitnessValues1);
            cudaFree(d_next_fitnessValues1);
            cudaFree(d_states1);
            cudaFree(d_distanceMatrix1);

            population1 = unflattenPopulation(flatPopulation1.data(), populationSize, numCities);
            int bestIndex = 0;
            for (int i = 0; i < populationSize; ++i) {
                double fit = host_fitness(cities, population1[i]);
                if (fit > bestFitness1) {
                    bestFitness1 = fit;
                    bestIndex = i;
                }
            }
            bestpath1 = population1[bestIndex];

        } else {

            cudaMemcpy(flatPopulation2.data(), d_population2, populationSize * numCities * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(d_cities2);
            cudaFree(d_population2);
            cudaFree(d_next_population2);
            cudaFree(d_fitnessValues2);
            cudaFree(d_next_fitnessValues2);
            cudaFree(d_states2);
            cudaFree(d_distanceMatrix2);

            population2 = unflattenPopulation(flatPopulation2.data(), populationSize, numCities);
            int bestIndex = 0;
            for (int i = 0; i < populationSize; ++i) {
                double fit = host_fitness(cities, population2[i]);
                if (fit > bestFitness2) {
                    bestFitness2 = fit;
                    bestIndex = i;
                }
            }
            bestpath2 = population2[bestIndex];
        }

        #pragma omp barrier
    }

    std::cout << "\nbestfitness 1: " << bestFitness1 << std::endl;
    std::cout << "best path 1 : " << 1 / bestFitness1 << std::endl;
    std::cout << "bestfitness 2: " << bestFitness2 << std::endl;
    std::cout << "best path 2 : " << 1 / bestFitness2 << std::endl;

    if(bestFitness1 >= bestFitness2){
        printf("return from GPU 0\n\n");
        return bestpath1;
    }else{
        printf("return from GPU 1\n\n");
        return bestpath2;
    } 
}

int main() {

    // std::vector<City> cities = {
    //     {60, 200}, {180, 200}, {80, 180}, {140, 180}, {20, 160} 
    //     // {100, 160}, {200, 160}, {140, 140}, {40, 120}, {100, 120},
    //     // {180, 100}, {60, 80}, {120, 80}, {180, 60}, {20, 40},
    //     // {100, 40}, {200, 40}, {20, 20}, {60, 20}, {160, 20}
    // };

    // std::string filename = "assets/qa194.tsp";
    // double answer = 9771.0;
    // std::string filename = "assets/uy734.tsp";
    // double answer = 86396.0;
    std::string filename = "assets/mg1000.tsp";
    double answer = 38185.0;

    std::vector<City> cities = readTSPFile(filename);
    std::cout << "\n====================================" << std::endl;
    std::cout << "map file        : " << filename << std::endl;

    auto start_time = std::chrono::steady_clock::now();
    int populationSize = cities.size() * 20;
    int generations = 4000;

    std::cout << "population size : " << populationSize << std::endl;
    std::cout << "generations     : " << generations << std::endl;

    std::vector<int> bestPath = geneticAlgorithm(cities, populationSize, generations);

    auto end_time = std::chrono::steady_clock::now();

    auto start = find(bestPath.begin(), bestPath.end(), 0);
    if (start != bestPath.end()) {
        size_t index = std::distance(bestPath.begin(), start);
        std::cout << "Best path: ";
        for (int i = 0; i < bestPath.size(); i++) {
            std::cout << bestPath[(index + i) % bestPath.size()] << " ";
        }
    }
    std::cout << "0" << std::endl;

    double totalDistance = host_pathDistance(cities, bestPath);

    std::cout << "Total distance: " << totalDistance << std::endl;
    std::cout << "error: " << answer / totalDistance * 100 << "%" << std::endl;
    std::cout<<"Running Times: "<< std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count() << " s\n";
    return 0;
}

/*

approximate best path from SOM-TSP:
    qa194   : 9700
    uy734   : 86000
    fi10639 : 638131
    it16862 : 

*/
