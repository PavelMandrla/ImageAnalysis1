//
// Created by pavel on 14.03.21.
//
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include "IALib/IALib.h"
#include "backprop.h"

using namespace std;
using namespace cv;

struct cl {
    int i;
    string name;
    std::vector<int> samples;
};

cl cl_box {0, "box", std::vector<int>{0,1,2,3} };
cl cl_star {1, "star", std::vector<int>{4,5,6,7} };
cl cl_rect {2, "rect", std::vector<int>{8,9,10,11} };

void train(NN * nn) {
    cv::Mat input = cv::imread( "../images/segmentation_input.png", cv::IMREAD_GRAYSCALE );
    doThresholding(128,input);

    cv::Mat indexed;
    int objectCount;
    indexObjects(input, indexed, objectCount);

    auto objects = calculateFeatures(objectCount, indexed);

    int n  = objectCount;
    double ** trainingSet = new double * [n];

    for (int i = 0; i < n; i++ ) {
        trainingSet[i] = new double[nn->n[0] + nn->n[nn->l - 1]];                                                       // number of input + number of output neurons
        trainingSet[i][0] = objects[i].x;
        trainingSet[i][1] = objects[i].y;

        trainingSet[i][nn->n[0]] = 0.0;
        trainingSet[i][nn->n[0] + 1] = 0.0;
        trainingSet[i][nn->n[0] + 2] = 0.0;
        if (std::find(cl_box.samples.begin(), cl_box.samples.end(), i) != cl_box.samples.end()) {   //IS BOX
            trainingSet[i][nn->n[0]] = 1.0;
        } else if (std::find(cl_star.samples.begin(), cl_star.samples.end(), i) != cl_star.samples.end()) { //IS STAR
            trainingSet[i][nn->n[0] + 1] = 1.0;
        } else { //IS RECT
            trainingSet[i][nn->n[0] + 2] = 1.0;
        }
    }

    double error = 1.0;
    int i = 0;
    while(error > 0.001) {
        setInput( nn, trainingSet[i%n] );
        feedforward( nn );
        error = backpropagation( nn, &trainingSet[i%n][nn->n[0]] );
        i++;
        printf( "\rerr=%0.3f", error );
    }
    printf( " (%d iterations)\n", i );

    for (int j = 0; j < n; j++) {
        delete [] trainingSet[j];
    }
    delete [] trainingSet;

}

void test(NN * nn) {
    cv::Mat detectInput = cv::imread( "../images/test01.png", cv::IMREAD_GRAYSCALE );
    cv::Mat detectInput_color = cv::imread( "../images/test01.png", cv::IMREAD_COLOR );
    doThresholding(128,detectInput);

    cv::Mat indexed;
    int objectCount;
    indexObjects(detectInput, indexed, objectCount);

    auto objects = calculateFeatures(objectCount, indexed);

    std::map<int, int> objectClass;

    auto* in = new double[nn->n[0]];
    for (int i = 0; i < objects.size(); i++) {
        in[0] = objects[i].x;
        in[1] = objects[i].y;

        setInput(nn, in, false);
        feedforward(nn);
        int output = getOutput(nn, false);
        objectClass.insert(pair<int, int>(i, output));
    }
    auto colored = colorObjects(indexed, objectCount, 3, objectClass, false);
    //imshow("result", colored);

    cv::Mat coloredScaled;
    cv::resize(colored, coloredScaled, cv::Size(), 2.0f, 2.0f, cv::INTER_NEAREST);

    cv::imshow("result", coloredScaled);
    cv::waitKey( 0 );
}

int main(int argc, char* argv[]) {
    NN * nn = createNN(2, 5, 3);
    train(nn);
    getchar();
    test(nn);
    releaseNN(nn);

    return 0;
}
