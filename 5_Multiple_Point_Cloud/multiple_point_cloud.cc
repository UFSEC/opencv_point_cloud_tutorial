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

// Filter the matches by an arbitrary distance
vector<DMatch> filter_matches(vector<DMatch> matches, int max_distance) {
  vector<DMatch> filtered_matches;
  for(auto match : matches) {
    //cout << "Distance: " << match.distance << endl;
    if(match.distance < max_distance) filtered_matches.push_back(match);
  }
  return filtered_matches;
}


int main() {
  // Create a viz window
  viz::Viz3d viz("Fun Visualization :)");

  string color_1_path = "../../../images/color.png";
  string depth_1_path = "../../../images/depth.png";
  string color_2_path = "../../../images/color_2.png";
  string depth_2_path = "../../../images/depth_2.png";

  Matx33d camera_matrix = getCameraMatrix();

  // Read in the images
  Mat img_color_1 = imread(color_1_path, CV_LOAD_IMAGE_COLOR);
  Mat img_depth_1 = imread(depth_1_path, CV_LOAD_IMAGE_ANYDEPTH);
  Mat img_color_2 = imread(color_2_path, CV_LOAD_IMAGE_COLOR);
  Mat img_depth_2 = imread(depth_2_path, CV_LOAD_IMAGE_ANYDEPTH);

  // Get the feature points
  Ptr<ORB> orb = ORB::create();
  vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;

  orb->detect(img_color_1, keypoints_1);
  orb->detect(img_color_2, keypoints_2);
  orb->compute(img_color_1, keypoints_1, descriptors_1);
  orb->compute(img_color_2, keypoints_2, descriptors_2);

  Mat img_keypoints1, img_keypoints2;
  drawKeypoints(img_color_1, keypoints_1, img_keypoints1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  drawKeypoints(img_color_2, keypoints_2, img_keypoints2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);








  // Match up the Feature Points between images
  BFMatcher matcher(NORM_L2);
  std::vector<DMatch> matches;
  matcher.match(descriptors_1, descriptors_2, matches);

  const int max_match_distance = 200;
  vector<DMatch> filtered_matches = filter_matches(matches, max_match_distance);
  
  Mat img_matches;
  drawMatches(img_color_1, keypoints_1, img_color_2, keypoints_2, filtered_matches, img_matches);
  //imshow("matches", img_matches);
  //waitKey(0);

  vector<Point> query_matched_keypoints_vec, train_matched_keypoints_vec;
  for(auto match: filtered_matches) {
    Point query_keypoint = keypoints_1[match.queryIdx].pt;
    Point train_keypoint = keypoints_2[match.trainIdx].pt;
    query_matched_keypoints_vec.push_back(query_keypoint);
    train_matched_keypoints_vec.push_back(train_keypoint);
    //cout << "Query Idx: " << match.queryIdx << ", Train Idx: " << match.trainIdx << endl;
    //cout << "Query Pt: " << query_keypoint << ", Train Pt: " << train_keypoint << endl;
  }
  Mat query_matched_keypoints(query_matched_keypoints_vec);
  Mat train_matched_keypoints(train_matched_keypoints_vec);

  // Select the matching keypoints and get their 3D representations
  Mat query_keypoints_3d, train_keypoints_3d;
  rgbd::depthTo3dSparse(img_depth_1, camera_matrix, query_matched_keypoints, query_keypoints_3d);
  rgbd::depthTo3dSparse(img_depth_2, camera_matrix, train_matched_keypoints, train_keypoints_3d);

  Mat filtered_query_3d, filtered_train_3d;

  cout << "query_keypoints_3d rows: " << query_keypoints_3d.rows << endl;
  cout << "train_keypoints_3d rows: " << train_keypoints_3d.rows << endl;
  for(int i = 0; i < query_keypoints_3d.rows; i++) {
    Vec3d query_vec = query_keypoints_3d.at<Vec3d>(i, 0);
    Vec3d train_vec = train_keypoints_3d.at<Vec3d>(i, 0);
    cout << "Query Vec: " << query_vec << ", Train Vec: " << train_vec << endl;

    double max_size = 1000000000000;
    double min_size = 1;
    if(query_vec[0] > max_size || query_vec[0] < min_size ||
       query_vec[1] > max_size || query_vec[1] < min_size ||
       query_vec[2] > max_size || query_vec[2] < min_size ||
       train_vec[0] > max_size || train_vec[0] < min_size ||
       train_vec[1] > max_size || train_vec[1] < min_size ||
       train_vec[2] > max_size || train_vec[2] < min_size) continue;
    cout << "" << endl;

    filtered_query_3d.push_back(query_vec);
    filtered_train_3d.push_back(train_vec);
  }

  cout << endl << endl << "Filtered!!!!" << endl << endl;

  cout << "filtered_query_3d rows: " << filtered_query_3d.rows << endl;
  cout << "filtered_train_3d rows: " << filtered_train_3d.rows << endl;
  for(int i = 0; i < filtered_query_3d.rows; i++) {
    Vec3d query_vec = filtered_query_3d.at<Vec3d>(i, 0);
    Vec3d train_vec = filtered_train_3d.at<Vec3d>(i, 0);
    cout << "Query Vec: " << query_vec << ", Train Vec: " << train_vec << endl;
  }








  // TODO: Estimate the affine 3d transform between images using matching keypoints
  std::vector<uchar> inliers;
  cv::Mat aff(3,4,CV_64F);
  int ret = cv::estimateAffine3D(filtered_query_3d, filtered_train_3d, aff, inliers);
  std::cout << "Transformation: " << endl << aff << std::endl;

  // This is necesssary to calculate the inverse of the affine transformation
  Mat homogenous_row = Mat(1, 4, CV_64F);
  homogenous_row.at<double>(0, 0) = 0;
  homogenous_row.at<double>(0, 1) = 0;
  homogenous_row.at<double>(0, 2) = 0;
  homogenous_row.at<double>(0, 3) = 1;
  aff.push_back(homogenous_row);

  Mat depth_1_cloud, depth_2_cloud;
  rgbd::depthTo3d(img_depth_1, camera_matrix, depth_1_cloud);
  rgbd::depthTo3d(img_depth_2, camera_matrix, depth_2_cloud);

  const int rows = 480;
  const int cols = 640;
  Mat depth_2_transformed(rows, cols, CV_64FC3);

  for(int i = 0; i < depth_2_cloud.rows; i++) {
    for(int j = 0; j < depth_2_cloud.cols; j++) {
      Vec3d my_3d = depth_2_cloud.at<Vec3d>(i, j);
      Mat mat41 = Mat::ones(4, 1, CV_64F);
      mat41.at<double>(0,0) = my_3d[0];
      mat41.at<double>(0,1) = my_3d[1];
      mat41.at<double>(0,2) = my_3d[2];
      Mat result41 = aff * mat41;
      Vec3d result_vec(result41.at<double>(0,0), result41.at<double>(0,1), result41.at<double>(0,2));
      depth_2_transformed.at<Vec3d>(i, j) = result_vec;
    }
  }
  
  Mat depth_combined, color_combined;
  hconcat(depth_1_cloud, depth_2_transformed, depth_combined);
  hconcat(img_color_1, img_color_2, color_combined);

  viz.showWidget("cloud", viz::WCloud(depth_combined, color_combined));
  viz.spin();

  waitKey(0);
  return 0;
}
