//
// Created by pavel on 19.04.21.
//

#include "Gaussian.h"
#include <cmath>

float Gaussian::pdf(float x) {
    float a = 1.0f / (sigma * sqrt(2 * M_PI));
    float exponent = - pow(x -this->mean, 2) / (2 * pow(this->sigma, 2));
    return a * std::exp(exponent);
}

float Gaussian::getWeightedPDF(float x) {
    return probability * this->pdf(x);
}

Gaussian::Gaussian(float mean, float sigma, float probability) {
    this->mean = mean;
    this->sigma = sigma;
    this->probability = probability;
}

void Gaussian::update(float x, float max_density, float sum) {
    this->probability = (1.0f - alpha) * probability + alpha * this->getWeightedPDF(x) / sum;
    double rho = (alpha * this->getWeightedPDF(x) / sum) / probability;
    this->mean = (1 - rho) * mean + rho * x;
    this->sigma = std::sqrt((1 - rho) * pow(sigma,2) + rho * pow(x - mean, 2));
}

MOG::MOG(int K) {
    this->K = K;
    float prob = 1.0f / float(K);
    float init_sigma = 30;

    for (int i = 0; i < K; i++) {
        this->gaussians.push_back(Gaussian(
                float(i+1) * 255.0f / float(K+1),
                init_sigma,
                prob));
    }
}

float MOG::getProbSum() {
    float probSum = 0.0;
    for (int k = 0; k < K; k++) {
        probSum += gaussians[k].probability;
    }
    return probSum;
}

void MOG::updateGaussians(float x, float max_density, float sum) {
    for (int k = 0; k < K; k++) {
        gaussians[k].update(x, max_density, sum);
    }
    this->normalizeProbabilities();
}

void MOG::normalizeProbabilities() {
    float probSum = this->getProbSum();
    for (int k = 0; k < K; k++) {
        gaussians[k].probability /= probSum;
    }
}

float MOG::getVal(float x) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += gaussians[k].getWeightedPDF(x);
    }
    return sum;
}

MOG_Matrix::MOG_Matrix(int width, int height, int K)
    : std::vector<std::vector<MOG>>(width, std::vector<MOG>(height, MOG(K))) {

}
