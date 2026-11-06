#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

Mat makeSkinMask(const Mat& frame) {
    Mat ycrcb, mask;
    cvtColor(frame, ycrcb, COLOR_BGR2YCrCb);
    mask = Mat::zeros(frame.size(), CV_8UC1);
    for (int y = 0; y < frame.rows; y++)
        for (int x = 0; x < frame.cols; x++) {
            Vec3b p = ycrcb.at<Vec3b>(y, x);
            int Y = p[0], Cr = p[1], Cb = p[2];
            if (Y > 40 && Y < 230 && Cr > 135 && Cr < 170 && Cb > 85 && Cb < 120)
                mask.at<uchar>(y, x) = 255;
        }
    morphologyEx(mask, mask, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
    dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
    morphologyEx(mask, mask, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
    return mask;
}
int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) { cout << "카메라 열기 실패" << endl; return -1; }

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        int W = frame.cols, H = frame.rows;
        int minPix = W * H * 3 / 1000;
        int maxPix = W * H * 40 / 100;
        Mat mask = makeSkinMask(frame);
        Mat labels, stats, centroids;
        int n = connectedComponentsWithStats(mask, labels, stats, centroids, 8);
        for (int i = 1; i < n; i++) {
            int area = stats.at<int>(i, CC_STAT_AREA);
            int x = stats.at<int>(i, CC_STAT_LEFT);
            int y = stats.at<int>(i, CC_STAT_TOP);
            int bw = stats.at<int>(i, CC_STAT_WIDTH);
            int bh = stats.at<int>(i, CC_STAT_HEIGHT);

            if (area < minPix || area > maxPix) continue;
            if (bw < 30 || bh < 30) continue;
            float ratio = (float)bw / bh;
            if (ratio < 0.4f || ratio > 2.0f) continue;

            Mat blob = (labels(Rect(x, y, bw, bh)) == i); 
            vector<vector<Point>> contours;
            vector<Vec4i> hier;
            findContours(blob.clone(), contours, hier, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
            int holeCount = 0, lowestHole = 0;
            for (size_t k = 0; k < contours.size(); k++) {
                if (hier[k][3] != -1) {                      
                    double a = contourArea(contours[k]);
                    if (a < bw * bh * 0.002) continue;        // 너무 작은 잡티 무시
                    holeCount++;
                    Rect hb = boundingRect(contours[k]);
                    int hy = hb.y + hb.height;                // 구멍 아래쪽 y 
                    if (hy > lowestHole) lowestHole = hy;
                }
            }
            if (holeCount < 2) continue;    // 눈·입 구멍이 있어야 얼굴 
            int chin = lowestHole + (int)(bh * 0.06);   // 입 아래로 살짝 = 턱
            if (chin > bh) chin = bh;
            rectangle(frame, Rect(x, y, bw, chin), Scalar(255, 0, 0), 2);
        }
        imshow("Face Detection", frame);
        imshow("Skin Mask", mask);
        if (waitKey(10) == 27) break;
    }
    return 0;
}