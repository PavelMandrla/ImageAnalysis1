//
// Created by pavel on 14.03.21.
//

#ifndef DZO_IALIB_H
#define DZO_IALIB_H

#include <opencv2/opencv.hpp>

template <typename T>
struct Coordinate {
    T x, y;
};

void doThresholding(int threshold, cv::Mat &image);

void floodFill(int y, int x, int I,cv::Mat &img, cv::Mat &indexed);

void indexObjects(cv::Mat &img, cv::Mat &indexed, int &objectCount);

cv::Mat colorObjects(cv::Mat indexed, int objCount, bool showIndices=true);

int getIndexAt(int y, int x, cv::Mat objects);

int getCircumference(int objectI, cv::Mat objects);

float getArea(int objectI, cv::Mat &objects);

Coordinate<int> getCenterOfMass(int objectI, cv::Mat objects);

float calculateF1(int objectI, cv::Mat &objects);

float calculateF2(int objectI, cv::Mat &objects);

std::vector<Coordinate<float>> calculateFeatures(int objectCount, cv::Mat &indexed);

#endif //DZO_IALIB_H
