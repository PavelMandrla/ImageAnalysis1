#include <opencv2/opencv.hpp>
#include <vector>
#include "IALib/Gaussian.h"

constexpr int K = 5;
constexpr float sigma_thresh = 0.006f;

void update(cv::Mat &frame, MOG_Matrix &mogMat, cv::Mat &mask) {
    #pragma omp parallel for
    for (int y = 0; y < frame.rows; y++) {
        for (int x = 0; x < frame.cols; x++) {
            uchar X = frame.at<uchar>(y, x);
            MOG &mog = mogMat[x][y];

            float densities[K];
            float densitySum = 0.0;

            int maximumIndex = 0;
            float densityMax = FLT_MIN;

            for (int k = 0; k < K; k++) {
                densities[k] = mog.gaussians[k].getWeightedPDF(X);
                densitySum += densities[k];

                if (densities[k] > densityMax) {
                    densityMax = densities[k];
                    maximumIndex = k;
                }
            }

            mog.updateGaussians(X, densityMax, densitySum);

            std::sort(mog.gaussians.begin(), mog.gaussians.end(), [](const Gaussian& g1, const Gaussian& g2) {
                return g1.probability > g2.probability;
            });

            mask.at<uchar>(y, x) = mog.getVal(X) < sigma_thresh ? 255 : 0;
        }
    }
}

bool getFrame(cv::VideoCapture &cap, cv::Mat &res) {
    cv::Mat tmpMat;
    if (!cap.read(tmpMat)) return false;

    std::vector<cv::Mat> channels(3);
    split(tmpMat, channels);

    cv::resize(channels[0], res, cv::Size(), 1, 1);
    return true;
}

int main(int argc, char* argv[]) {
    cv::VideoCapture cap("../images/dt_passat.mpg");
    cv::Mat frame;

    if (!getFrame(cap, frame)) return 1;
    MOG_Matrix mogs(frame.cols, frame.rows, K);
    cv::Mat mask = frame.clone();

    do {
        update(frame, mogs, mask);

        cv::imshow("hello", frame);
        cv::imshow("mask", mask);

        cv::waitKey(10);
    } while (getFrame(cap, frame));

    cap.release();
    return 0;
}