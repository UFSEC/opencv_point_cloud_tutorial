#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/rgbd.hpp>
#include <iostream>
#include <string>
#include <vector>

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
  string color2_path = "../../../images/color_2.png";
  string depth2_path = "../../../images/depth_2.png";

  // Read in the images
  Mat color = imread(color_path, CV_LOAD_IMAGE_COLOR);
  Mat depth = imread(depth_path, CV_LOAD_IMAGE_ANYDEPTH);

  Matx33d camera_matrix = getCameraMatrix();

  Mat depth_cloud;
  rgbd::depthTo3d(depth, camera_matrix, depth_cloud);


  Mat color2 = imread(color2_path, CV_LOAD_IMAGE_COLOR);
  Mat depth2 = imread(depth2_path, CV_LOAD_IMAGE_ANYDEPTH);

  imshow("color2", color2);
  imshow("depth2", depth2);

  Mat depth2_cloud;
  rgbd::depthTo3d(depth2, camera_matrix, depth2_cloud);

  cout << "depth cloud rows: " << depth_cloud.rows << ", cols: " << depth_cloud.cols << endl;
  cout << "depth cloud 2 rows: " << depth2_cloud.rows << ", cols: " << depth2_cloud.cols << endl;

  vector<Point3d> first, second;

  for(int i = 0; i < depth_cloud.rows; i++) {
    for(int j = 0; j < depth_cloud.cols; j++) {
      Vec3d vec = depth_cloud.at<Vec3d>(i, j);
      Point3d point(vec);
      first.push_back(point);
    }
  }
  for(int i = 0; i < depth2_cloud.rows; i++) {
    for(int j = 0; j < depth2_cloud.cols; j++) {
      Vec3d vec = depth2_cloud.at<Vec3d>(i, j);
      Point3d point(vec);
      second.push_back(point);
    }
  }

  std::vector<uchar> inliers;
  cv::Mat aff(4,4,CV_64F);

  int ret = cv::estimateAffine3D(first, second, aff, inliers);
  Mat homogenous_row = Mat(1, 4, CV_64F);
  homogenous_row.at<double>(0, 0) = 0;
  homogenous_row.at<double>(0, 1) = 0;
  homogenous_row.at<double>(0, 2) = 0;
  homogenous_row.at<double>(0, 3) = 1;
  aff.push_back(homogenous_row);

  std::cout << "Transformation: " << aff << std::endl;

  Mat depth2_transformed(480, 640, CV_64FC3);

  for(int i = 0; i < depth2_cloud.rows; i++) {
    for(int j = 0; j < depth2_cloud.cols; j++) {
      Vec3d my_3d = depth2_cloud.at<Vec3d>(i, j);
      Mat mat41 = Mat(4, 1, CV_64F);
      mat41.at<double>(0,0) = my_3d[0];
      mat41.at<double>(1,0) = my_3d[1];
      mat41.at<double>(2,0) = my_3d[2];
      mat41.at<double>(3,0) = 1.0;

      Mat result41 = aff.inv() * mat41;
      Vec3d result_vec(result41.at<double>(0,0), result41.at<double>(1,0), result41.at<double>(2,0));
      depth2_transformed.at<Vec3d>(i, j) = result_vec;
    }
  }
  
  Mat depth_combined, color_combined;
  hconcat(depth_cloud, depth2_transformed, depth_combined);
  hconcat(color, color2, color_combined);

  viz.showWidget("cloud", viz::WCloud(depth_combined, color_combined));
  viz.spin();

  waitKey(0);
  return 0;
}
