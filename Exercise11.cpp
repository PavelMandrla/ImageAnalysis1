#include <opencv2/opencv.hpp>
#include <vector>

constexpr int K_x = 4;
constexpr int K_y = 3;
double K = K_x * K_y;

constexpr int m = 50;
double N;
double S;

struct Center {
    int x, y;
};

bool operator==(const Center& lhs, const Center& rhs) {
    return lhs.x==rhs.x && lhs.y==rhs.y;
}

bool operator!=(const Center& lhs, const Center& rhs) {
    return !(lhs == rhs);
}

std::vector<Center> placeCenters(cv::Mat &img) {
    int dx = img.cols / (K_x);
    int dy = img.rows / (K_y);
    std::vector<Center> result;
    for (int k_x = dx/2; k_x < img.cols; k_x += dx) {
        for (int k_y = dy/2; k_y < img.rows; k_y += dy) {
            //img.at<cv::Vec3b>(k_y, k_x) = cv::Vec3b (0,0,255);

            Center tmp;
            tmp.x = k_x;
            tmp.y = k_y;
            result.push_back(tmp);
        }
    }
    return result;
}

uchar getValAt(cv::Mat &laplac, int x, int y) {
    if (x < 0 || x >= laplac.cols || y < 0 || y >= laplac.rows) {
        return 255;
    }
    return laplac.at<uchar>(y, x);
}

Center getLowestInArea(cv::Mat &laplac, Center &center) {
    int lx = 0;
    int ly = 0;
    uchar lval = getValAt(laplac, center.x, center.y);
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            uchar tmpVal = getValAt(laplac, center.x + dx, center.y + dy);
            if (tmpVal < lval) {
                lval = tmpVal;
                lx = dx;
                ly = dy;
            }
        }
    }
    return Center{center.x + lx, center.y + ly};
}

bool moveCenter(cv::Mat &laplac, Center &center) {
    Center newPos = getLowestInArea(laplac, center);
    if (center != newPos) {
        center = newPos;
        return true;
    }
    return false;
}

void moveCenters(cv::Mat &img, std::vector<Center> &centers) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::Mat laplac;
    cv::Laplacian(gray, laplac, CV_8UC1);

    bool moved = true;
    while (moved) {
        moved = false;
        for (int i = 0; i < centers.size(); i++) {
            moved |= moveCenter(img, centers[i]);
            //img.at<cv::Vec3b>(centers[i].y, centers[i].x) = cv::Vec3b(0,0,255);
        }
        //cv::imshow("fml", img);
        //cv::waitKey(10);
    }
}


double dRGB(cv::Mat &img, int x1, int y1, int x2, int y2) {
    cv::Vec3b c1 = img.at<cv::Vec3b>(y1, x1);
    cv::Vec3b c2 = img.at<cv::Vec3b>(y2, x2);

    return sqrt(pow(c2[0] - c1[0], 2) + pow(c2[1] - c1[1], 2) + pow(c2[2] - c1[2], 2));
}

double dXY(int x1, int y1, int x2, int y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

double D_s(cv::Mat &img, int x1, int y1, int x2, int y2) {
    return dRGB(img, x1, y1, x2, y2) + (m/S) * dXY(x1, y1, x2, y2);
}

Center getClosestCenter(cv::Mat &img, int x, int y, const std::vector<Center>& centers) {
    Center res = centers[0];
    double resD = D_s(img, x, y, res.x, res.y);
    for (auto center : centers) {
        double tmpD = D_s(img, x, y, center.x, center.y);
        if (tmpD < resD) {
            res = center;
            resD = tmpD;
        }
    }
    return res;
}

cv::Mat slic(cv::Mat &img, const std::vector<Center>& centers) {
    cv::Mat result(img.size(), img.type());
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            auto c = getClosestCenter(img, x, y, centers);
            result.at<cv::Vec3b>(y, x) = img.at<cv::Vec3b>(c.y, c.x);
        }
    }
    return result;
}

int main() {
    cv::Mat input = cv::imread("../images/bear.png", cv::IMREAD_COLOR);
    N = input.rows * input.cols;
    S = sqrt(N / K);

    auto centers = placeCenters(input);
    moveCenters(input, centers);
    cv::imshow("test",input);

    cv::imshow("result", slic(input, centers));
    cv::waitKey(0);





    return 0;
}
