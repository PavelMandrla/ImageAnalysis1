//
// Created by pavel on 19.04.21.
//

#ifndef IMAGEANALYSIS1_GAUSSIAN_H
#define IMAGEANALYSIS1_GAUSSIAN_H

#include <vector>

constexpr float lambda = 1.5f;
constexpr float alpha = 0.01f;

class Gaussian {
public:
    float mean;
    float sigma;
    float probability;

    Gaussian(float mean, float sigma, float probability);

    float pdf(float x);
    float getWeightedPDF(float x);
    void update(float x, float maxDensity, float densitySum);
};

class MOG {
public:
    std::vector<Gaussian> gaussians;
    int K;

    explicit MOG(int K);

    float getProbSum();
    void updateGaussians(float x, float max_density, float sum);
    void normalizeProbabilities();
    float getVal(float x);
};

class MOG_Matrix : public std::vector<std::vector<MOG>> {
public:
    MOG_Matrix(int width, int height, int K);
};


#endif //IMAGEANALYSIS1_GAUSSIAN_H
