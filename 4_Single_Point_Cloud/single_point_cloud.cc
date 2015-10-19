#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/rgbd.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

Matx33d getCameraMatrix() {
  // These are the hardcoded camera intrinsics values
  float fx = 532.189488;
  float fy = 532.189488;
  float cx = 318.839986;
  float cy = 244.149197;
  float scale = 10000.0;

  return Matx33d( fx, 0.0, cx,
                  0.0, fy, cy,
                  0.0, 0.0, 1.0);
}

int main() {
  // Create a viz window
  viz::Viz3d viz("Coordinate Frame");

  string color_path = "../../../images/color.png";
  string depth_path = "../../../images/depth.png";

  // Read in the images
  Mat color = imread(color_path, CV_LOAD_IMAGE_COLOR);
  Mat depth = imread(depth_path, CV_LOAD_IMAGE_ANYDEPTH);

  Matx33d camera_matrix = getCameraMatrix();

  Mat depth_cloud;
  rgbd::depthTo3d(depth, camera_matrix, depth_cloud);

  viz.showWidget("cloud", viz::WCloud(depth_cloud, color));
  viz.spin();

  waitKey(0);
  return 0;
}
