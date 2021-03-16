//
// Created by pavel on 14.03.21.
//

#include "IALib.h"
#include <cmath>
#include <random>
#include <string>

void doThresholding(int threshold, cv::Mat &image) {
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            image.at<uchar>(y, x) = image.at<uchar>(y,x) > threshold ? 255 : 0 ;
        }
    }
}

void floodFill(int y, int x, int I,cv::Mat &img, cv::Mat &indexed) {
    if (y < 0 || y >= img.rows || x < 0 || x >= img.cols) return;
    if (!img.at<uchar>(y, x)) return;
    if (indexed.at<int>(y, x) != 0) return;
    indexed.at<int>(y, x) = I;
    floodFill(y-1, x, I, img, indexed);
    floodFill(y+1, x, I, img, indexed);
    floodFill(y, x-1, I, img, indexed);
    floodFill(y, x+1, I, img, indexed);
    floodFill(y-1, x+1, I, img, indexed);
    floodFill(y+1, x+1, I, img, indexed);
    floodFill(y-1, x-1, I, img, indexed);
    floodFill(y+1, x-1, I, img, indexed);
}

void indexObjects(cv::Mat &img, cv::Mat &indexed, int &objectCount) {
    indexed = cv::Mat::zeros(img.size(), CV_32SC1);

    objectCount = 0;
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            if (!img.at<uchar>(y, x) || indexed.at<int>(y, x)) continue;
            objectCount++;
            floodFill(y, x, objectCount, img, indexed);
        }
    }

    //SET BACKGROUND TO -1
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            indexed.at<int>(y, x) -= 1;
        }
    }
    objectCount;
}

cv::Mat colorObjects(cv::Mat indexed, int objCount, bool showIndices) {
    auto *colors = new cv::Vec3i[objCount];

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distrib(40, 255);

    for (int i = 0; i < objCount; i++) {
        colors[i] = cv::Vec3i { distrib(gen), distrib(gen), distrib(gen) };
    }

    cv::Mat colored = cv::Mat::zeros(indexed.size(), CV_8SC3);

    for (int y = 0; y < colored.rows; y++) {
        for (int x = 0; x < colored.cols; x++) {
            if (indexed.at<int>(y, x) != -1) {
                colored.at<cv::Vec3b>(y, x) = colors[indexed.at<int>(y, x)];
            }
        }
    }

    if (showIndices) {
        for (int i = 0; i < objCount; i++) {
            Coordinate<int> center = getCenterOfMass(i, indexed);
            cv::putText(
                    colored,
                    std::to_string(i),
                    cv::Point(center.x, center.y),
                    cv::FONT_HERSHEY_DUPLEX,
                    0.5,
                    cv::Scalar(0,255,0),
                    2,
                    false);
        }
    }

    delete[] colors;
    return colored;
}

int getIndexAt(int y, int x, cv::Mat objects) {
    if (y < 0 || x < 0 || y >= objects.rows || x >= objects.cols) {
        return -1;
    }
    return objects.at<int>(y-1, x);
}

int getCircumference(int objectI, cv::Mat objects) {
    int result = 0;
    for (int y = 0; y < objects.rows; y++) {
        for (int x = 0; x < objects.cols; x++) {
            if (objects.at<int>(y, x) != objectI) continue;

            if (getIndexAt(y+1, x, objects) == objectI &&
                getIndexAt(y-1, x, objects) == objectI &&
                getIndexAt(y, x+1, objects) == objectI &&
                getIndexAt(y, x-1, objects) == objectI
                    ) continue;
            result++;
        }
    }
    return result;
}

float m(int p, int q, int objectI, cv::Mat &objects) {
    float result = 0;
    for (int y = 0; y < objects.rows; y++) {
        for (int x = 0; x < objects.cols; x++) {
            if (objects.at<int>(y, x) != objectI) continue;
            result += pow(x, p) * pow(y, q);
        }
    }
    return result;
}

float getArea(int objectI, cv::Mat &objects) {
    return m(0, 0, objectI, objects);
}

Coordinate<int> getCenterOfMass(int objectI, cv::Mat objects) {
    return Coordinate<int>{
            int(m(1, 0, objectI, objects) / m(0, 0, objectI, objects)),
            int(m(0, 1, objectI, objects) / m(0, 0, objectI, objects))
    };
}

float calculateF1(int objectI, cv::Mat &objects) {
    return pow(getCircumference(objectI, objects),2) / (100 * getArea(objectI, objects));
}

float u(int p, int q, int objectI, cv::Mat &objects) {
    float result = 0;
    auto center = getCenterOfMass(objectI, objects);
    for (int y = 0; y < objects.rows; y++) {
        for (int x = 0; x < objects.cols; x++) {
            if (objects.at<int>(y, x) != objectI) continue;
            result += pow(x - center.x, p) * pow(y - center.y, q);
        }
    }
    return result;
}

float calculateF2(int objectI, cv::Mat &objects) {
    float u2_0 = u(2, 0, objectI, objects);
    float u0_2 = u(0, 2, objectI, objects);
    float a = (u2_0 + u0_2) / 2;
    float b = sqrt(4 * pow(u(1,1, objectI, objects), 2) + pow(u2_0 - u0_2, 2)) / 2;

    float u_min = a + b;
    float u_max = a - b;
    return u_min / u_max;
}

std::vector<Coordinate<float>> calculateFeatures(int objectCount, cv::Mat &indexed) {
    std::vector<Coordinate<float>> result;
    for (int i = 0; i < objectCount; i++) {
        result.push_back({
            calculateF1(i, indexed),
            calculateF2(i, indexed)
        });
    }
    return result;
}