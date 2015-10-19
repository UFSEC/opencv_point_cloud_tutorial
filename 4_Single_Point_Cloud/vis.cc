#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

/**
 * @function main
 */
int main()
{
    /// Create a window
    viz::Viz3d myWindow("Coordinate Frame");

    /// Add coordinate axes
    myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

    string image1_path = "../../../images/color.png";
    string image2_path = "../../../images/depth.png";

    string window1_title = "color";
    string window2_title = "depth";

    namedWindow(window1_title, WINDOW_AUTOSIZE);
    namedWindow(window2_title, WINDOW_AUTOSIZE);

    // Read in the images
    Mat image1 = imread(image1_path);
    Mat image2 = imread(image2_path);

    imshow(window1_title, image1);
    imshow(window2_title, image2);

    waitKey(0);
    return 0;
    /*
    // Create the camera calibration matrix
    camera_matrix = np.zeros((3,3))
    fx = 532.189488 
    fy = 532.189488 
    cx = 318.839986 
    cy = 244.149197
    scale = 10000.0
    */

    /*
    /// Add line to represent (1,1,1) axis
    viz::WLine axis(Point3f(-1.0f,-1.0f,-1.0f), Point3f(1.0f,1.0f,1.0f));
    axis.setRenderingProperty(viz::LINE_WIDTH, 4.0);
    myWindow.showWidget("Line Widget", axis);

    /// Construct a cube widget
    viz::WCube cube_widget(Point3f(0.5,0.5,0.0), Point3f(0.0,0.0,-0.5), true, viz::Color::blue());
    cube_widget.setRenderingProperty(viz::LINE_WIDTH, 4.0);

    /// Display widget (update if already displayed)
    myWindow.showWidget("Cube Widget", cube_widget);

    /// Rodrigues vector
    Mat rot_vec = Mat::zeros(1,3,CV_32F);
    float translation_phase = 0.0, translation = 0.0;
    while(!myWindow.wasStopped())
    {
        // Rotation using rodrigues
        /// Rotate around (1,1,1)
        rot_vec.at<float>(0,0) += CV_PI * 0.01f;
        rot_vec.at<float>(0,1) += CV_PI * 0.01f;
        rot_vec.at<float>(0,2) += CV_PI * 0.01f;

        /// Shift on (1,1,1)
        translation_phase += CV_PI * 0.01f;
        translation = sin(translation_phase);

        Mat rot_mat;
        Rodrigues(rot_vec, rot_mat);

        /// Construct pose
        Affine3f pose(rot_mat, Vec3f(translation, translation, translation));

        myWindow.setWidgetPose("Cube Widget", pose);

        myWindow.spinOnce(1, true);
    }
    */
}
