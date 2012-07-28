#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

int clipLimit = 1;
int nTilesX = 8;
int nTilesY = 8;
int nBins = 256;

Mat image, gray, adjusted;

// define a trackbar callback
static void onTrackbar(int, void*)
{
    try {
        adapthisteq(gray, adjusted, nTilesX, nTilesY, clipLimit * 0.01, nBins);
    } catch (Exception& e) {
        std::cout << e.what();
        return;
    }
    imshow("CLAHE", adjusted);
}

static void help()
{
    printf("\nThis sample demonstrates Contrast limited adaptive histogram equalization (CLAHE)\n"
           "Call:\n"
           "    /.adapthisteq [image_name -- Default is fruits.jpg]\n\n");
}

const char* keys =
{
    "{1| |fruits.jpg|input image name}"
};

const char windowTitle[] = "Contrast limited adaptive histogram equalization (CLAHE)";

int main( int argc, const char** argv )
{
    help();

    CommandLineParser parser(argc, argv, keys);
    string filename = parser.get<string>("1");

    image = imread(filename, 1);
    if(image.empty())
    {
        printf("Cannot read image file: %s\n", filename.c_str());
        help();
        return -1;
    }

    cvtColor(image, gray, CV_BGR2GRAY);

    // Create a window
    namedWindow(windowTitle, 1);

    // create a toolbar
    createTrackbar("nTilesX", windowTitle, &nTilesX, 16, onTrackbar);
    createTrackbar("nTilesY", windowTitle, &nTilesY, 16, onTrackbar);
    createTrackbar("clipLimit", windowTitle, &clipLimit, 2000, onTrackbar);
    createTrackbar("nBins", windowTitle, &nBins, 256, onTrackbar);

    // Show the image
    onTrackbar(0, 0);

    // Wait for a key stroke; the same function arranges events processing
    waitKey(0);

    return 0;
}
