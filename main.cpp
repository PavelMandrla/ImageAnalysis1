#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <random>
#include <vector>
#include "backprop.h"

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
}

cv::Mat colorObjects(cv::Mat indexed, int objCount) {
    cv::Vec3i *colors = new cv::Vec3i[objCount];

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distrib(40, 255);

    for (int i = 0; i < objCount; i++) {
        colors[i] = cv::Vec3i { distrib(gen), distrib(gen), distrib(gen) };
    }

    cv::Mat colored = cv::Mat::zeros(indexed.size(), CV_8SC3);

    for (int y = 0; y < colored.rows; y++) {
        for (int x = 0; x < colored.cols; x++) {
            if (indexed.at<int>(y, x)) {
                colored.at<cv::Vec3b>(y, x) = colors[indexed.at<int>(y, x)];
            }
        }
    }

    delete[] colors;
    return colored;
}

void cv1() {
    cv::Mat input = cv::imread( "../images/test02.png", cv::IMREAD_GRAYSCALE );
    doThresholding(128,input);

    cv::Mat indexed;
    int objectCount;
    indexObjects(input, indexed, objectCount);

    cv::Mat colored = colorObjects(indexed, objectCount);
    cv::Mat coloredScaled;
    cv::resize(colored, coloredScaled, cv::Size(), 4.0f, 4.0f, cv::INTER_NEAREST);
    cv::imshow("result", coloredScaled);
    cv::waitKey( 0 );
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
   return m(0,0, objectI, objects);
}

struct Coordinate {
    int x;
    int y;
};

Coordinate getCenterOfMass(int objectI, cv::Mat objects) {
    return Coordinate{
        int(m(1, 0, objectI, objects) / m(0, 0, objectI, objects)),
        int(m(0, 1, objectI, objects) / m(0, 0, objectI, objects))
    };
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

float calculateF1(int objectI, cv::Mat &objects) {
    return pow(getCircumference(objectI, objects),2) / (100 * getArea(objectI, objects));
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

void cv2() {
    cv::Mat input = cv::imread( "../images/segmentation_input.png", cv::IMREAD_GRAYSCALE );
    doThresholding(128,input);

    cv::Mat indexed;
    int objectCount;
    indexObjects(input, indexed, objectCount);

    cv::Mat colored = colorObjects(indexed, objectCount);
    cv::Mat coloredScaled;
    cv::resize(colored, coloredScaled, cv::Size(), 4.0f, 4.0f, cv::INTER_NEAREST);

    for (int i = 1; i < objectCount; i++) {
        printf("Object %d\n\tArea: %f\n\tCircumference: %d\n", i, getArea(i, indexed), getCircumference(i, indexed));
        printf("\tF1: %f\n\tF2: %f\n", i, calculateF1(i, indexed), calculateF2(i, indexed));
    }

    cv::imshow("result", coloredScaled);
    cv::waitKey( 0 );
}

struct Coordinate_d {
    double x;
    double y;
};

Coordinate_d computeEthalon(std::vector<int> indexes,cv::Mat &objects) {
    Coordinate_d ethalon { 0.0, 0.0 };
    for (auto i : indexes) {
        ethalon.x += double(calculateF1(i, objects));
        ethalon.y += double(calculateF2(i, objects));
    }
    ethalon.x /= (double) indexes.size();
    ethalon.y /= (double) indexes.size();
    return ethalon;
}

int findClosestEthalon(Coordinate_d center, std::vector<Coordinate_d> ethalons ) {
    std::vector<double> distances;
    for (auto ethalon : ethalons) {
        distances.push_back(sqrt(pow((double) center.x - ethalon.x, 2) + pow((double) center.y - ethalon.y, 2)));
    }
    printf("%f,%f\n", center.x, center.y);


    int closestI = 0;
    double closestDist = distances[0];
    for (int i = 1; i < distances.size(); i++) {
        if (distances[i] < closestDist) {
           closestI = i;
           closestDist = distances[i];
        }
    }
    return closestI;
}

void cv3() {
    cv::Mat input = cv::imread( "../images/segmentation_input.png", cv::IMREAD_GRAYSCALE );
    doThresholding(128,input);

    cv::Mat indexed;
    int objectCount;
    indexObjects(input, indexed, objectCount);

    std::vector<int> boxIs = {1,2,3,4};
    std::vector<int> starIs = {5,6,7,8};
    std::vector<int> rectIs = {9,10,11,12};

    for (int i = 1; i <= objectCount; i++) {
        printf("%f,%f\n", double(calculateF1(i, indexed)), double(calculateF2(i, indexed)));
    }

    auto boxEthalon     = computeEthalon(std::vector<int>{1,2,3,4}, indexed);
    auto starEthalon    = computeEthalon(std::vector<int>{5,6,7,8}, indexed);
    auto rectEthalon    = computeEthalon(std::vector<int>{9,10,11,12}, indexed);

    printf("\n%f,%f\n", boxEthalon.x, boxEthalon.y);
    printf("%f,%f\n", starEthalon.x, starEthalon.y);
    printf("%f,%f\n\n", rectEthalon.x, rectEthalon.y);

    cv::Mat detectInput = cv::imread( "../images/test01.png", cv::IMREAD_GRAYSCALE );
    cv::Mat detectInput_color = cv::imread( "../images/test01.png", cv::IMREAD_COLOR );
    doThresholding(128,detectInput);
    cv::Mat detectIndexed;
    int detectObjectCount;
    indexObjects(detectInput, detectIndexed, detectObjectCount);

    std::vector<Coordinate_d> ethalons = { boxEthalon, starEthalon, rectEthalon };
    for (int i = 1; i <= detectObjectCount; i++) {
        Coordinate center = getCenterOfMass(i, detectIndexed);
        Coordinate_d objEth = {
                double(calculateF1(i, detectIndexed)),
                double(calculateF2(i, detectIndexed))
        };
        switch (findClosestEthalon(objEth, ethalons)) {
            case 0: {
                cv::putText(detectInput_color, "box", cv::Point(center.x, center.y),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,0),2,false);
                break;
            }
            case 1: {
                cv::putText(detectInput_color, "star", cv::Point(center.x, center.y),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(255,255,0),2,false);
                break;
            }
            case 2: {
                cv::putText(detectInput_color, "rect", cv::Point(center.x, center.y),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,255),2,false);
                break;
            }
        }
    }

    cv::Mat coloredScaled;
    cv::resize(detectInput_color, coloredScaled, cv::Size(), 2.0f, 2.0f, cv::INTER_NEAREST);

    cv::imshow("result", coloredScaled);
    cv::waitKey( 0 );
}

void getBounds(std::vector<Coordinate_d> &objects, Coordinate_d &from, Coordinate_d &to) {
    from = objects[0];
    to = objects[0];

    for (auto o : objects) {
        if (o.x < from.x) from.x = o.x;
        if (o.y < from.y) from.y = o.y;
        if (o.x > to.x) to.x = o.x;
        if (o.x > to.x) to.x = o.y;
    }
}

std::vector<Coordinate_d> initCentroids(int k, std::vector<Coordinate_d> &objects) {
    Coordinate_d boundsFrom{}, boundsTo{};
    getBounds(objects, boundsFrom, boundsTo);

    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_real_distribution<double> x(boundsFrom.x, boundsTo.x);
    std::uniform_real_distribution<double> y(boundsFrom.y, boundsTo.y);

    std::vector<Coordinate_d> result;
    for (int i = 0; i < k; i++) {
        result.push_back({ x(mt), y(mt) });
    }
    return result;
}

double dst(Coordinate_d a, Coordinate_d b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

Coordinate_d calculateCentroid(std::vector<int> &indicies, std::vector<Coordinate_d> &objects) {
    Coordinate_d result {0, 0};
    for (auto i : indicies) {
        result.x += objects[i].x;
        result.y += objects[i].y;
    }
    result.x /= indicies.size();
    result.y /= indicies.size();

    return result;
}

std::vector<Coordinate_d> adjustCentroids(std::vector<Coordinate_d> &centroids, std::vector<Coordinate_d> &objects) {
    std::vector<std::vector<int>> ownedIndicies(centroids.size());

    for (int i = 0; i < objects.size(); i++) {
        int clJ = 0;
        double clDist = dst(centroids[clJ], objects[i]);
        for (int j = 1; j < centroids.size(); j++) {
            double dist = dst(centroids[j], objects[i]);
            if (dist < clDist) {
                clJ = j;
                clDist = dist;
            }
        }
        ownedIndicies[clJ].push_back(i);
    }

    std::vector<Coordinate_d> nCentroids;
    for (auto oI : ownedIndicies) {
        nCentroids.push_back(calculateCentroid(oI, objects));
    }
    return nCentroids;
}

double getMaxShift(std::vector<Coordinate_d> &oldCentroids, std::vector<Coordinate_d> &newCentroids) {
    double maxShift = dst(oldCentroids[0], newCentroids[0]);
    for (int i = 1; i < oldCentroids.size(); i++) {
        auto d = dst(oldCentroids[i], newCentroids[i]);
        if (d > maxShift) {
            maxShift = d;
        }
    }
    return maxShift;
}

std::vector<Coordinate_d>  kMeans(int k, double minShift, std::vector<Coordinate_d> &objects) {
    auto centroids = initCentroids(k, objects);

    double shift = 100;
    while (shift > minShift) {
        auto nCentroids = adjustCentroids(centroids, objects);
        shift = getMaxShift(centroids, nCentroids);
        centroids = nCentroids;
    }
    return centroids;
}

std::vector<Coordinate_d> calculateFeatures(int objectCount, cv::Mat &indexed) {
    std::vector<Coordinate_d> result;
    for (int i = 1; i <= objectCount; i++) {
        result.push_back({
            double(calculateF1(i, indexed)),
            double(calculateF2(i, indexed))
        });
    }
    return result;
}

void cv4() {
    cv::Mat input = cv::imread( "../images/segmentation_input.png", cv::IMREAD_GRAYSCALE );
    doThresholding(128,input);

    cv::Mat indexed;
    int objectCount;
    indexObjects(input, indexed, objectCount);

    auto objects = calculateFeatures(objectCount, indexed);
    auto centroids = kMeans(3, 0.1, objects);

    cv::Mat detectInput = cv::imread( "../images/test01.png", cv::IMREAD_GRAYSCALE );
    cv::Mat detectInput_color = cv::imread( "../images/test01.png", cv::IMREAD_COLOR );
    doThresholding(128,detectInput);
    cv::Mat detectIndexed;
    int detectObjectCount;
    indexObjects(detectInput, detectIndexed, detectObjectCount);

    for (int i = 1; i <= detectObjectCount; i++) {
        Coordinate center = getCenterOfMass(i, detectIndexed);
        Coordinate_d objFeatures = {
                double(calculateF1(i, detectIndexed)),
                double(calculateF2(i, detectIndexed))
        };
        cv::putText(
                detectInput_color,
                std::to_string(findClosestEthalon(objFeatures, centroids)),
                cv::Point(center.x, center.y),
                cv::FONT_HERSHEY_DUPLEX,
                1,
                cv::Scalar(0,255,0),
                2,
                false);
    }
    cv::imshow("result", detectInput_color);
    cv::waitKey( 0 );
}

void train(NN* nn) {
    int n = 1000;
    double ** trainingSet = new double * [n];
    for (int i = 0; i < n; i++) {
        trainingSet[i] = new double[nn->n[0] + nn->n[nn->l - 1]];

        bool classA = i % 2;

        for ( int j = 0; j < nn->n[0]; j++ ) {
            if ( classA ) {
                trainingSet[i][j] = 0.1 * ( double )rand() / ( RAND_MAX ) + 0.6;
            } else {
                trainingSet[i][j] = 0.1 * ( double )rand() / ( RAND_MAX ) + 0.2;
            }
        }

        trainingSet[i][nn->n[0]] = ( classA )? 1.0 : 0.0;
        trainingSet[i][nn->n[0] + 1] = ( classA )? 0.0 : 1.0;
    }

    double error = 1.0;
    int i = 0;
    while(error > 0.001)
    {
        setInput( nn, trainingSet[i%n] );
        feedforward( nn );
        error = backpropagation( nn, &trainingSet[i%n][nn->n[0]] );
        i++;
        printf( "\rerr=%0.3f", error );
    }
    printf( " (%d iterations)\n", i );

    for ( int i = 0; i < n; i++ ) {
        delete [] trainingSet[i];
    }
    delete [] trainingSet;
}

void cv5() {
    NN * nn = createNN(2, 4, 2);
    train(nn);

}

int main() {
    //cv1();
    //cv2();
    //cv3();
    //cv4();

    return 0;
}
