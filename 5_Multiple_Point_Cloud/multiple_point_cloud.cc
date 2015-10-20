#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Geometry>

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

struct KeypointFrame {
 public:
  cv::Mat rgb_image;
  cv::Mat depth_image;
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  std::vector<Eigen::Vector3f> points;
};

void ReprojectKeypoints(const std::vector<cv::KeyPoint>& keypoints,
                        const cv::Mat& depth,
                        const cv::Matx33d& intr,
                        std::vector<Eigen::Vector3f>* points) {
  for (std::size_t i = 0; i < keypoints.size(); ++i) {
    float x, y, z;
    cv::Point point = keypoints[i].pt;
    z = depth.at<float>(point.y, point.x);
    x = (point.x - intr(0,2)) / intr(0,0) * z;
    y = (point.y - intr(1,2)) / intr(1,1) * z;
    points->emplace_back(x, y, z);
  }
}

void Gen3RandomIndices(const int max, int* a, int* b, int* c) {
       *a = rand() % max;
  do { *b = rand() % max; } while (*b == *a);
  do { *c = rand() % max; } while (*c == *b || *c == *a);
}

Eigen::Vector3f TransformPoint(const Eigen::Vector3f& point,
                               const Eigen::Isometry3f& transformation) {
  Eigen::Vector4f point_prime(point.x(), point.y(), point.z(), 1);
  Eigen::Vector4f transformed_point_prime = transformation * point_prime;
  Eigen::Vector3f transformed_point(transformed_point_prime.x(),
                                    transformed_point_prime.y(),
                                    transformed_point_prime.z());
  return transformed_point;
}

bool RefineInliersUmeyama(const KeypointFrame& from_frame,
                          const KeypointFrame& to_frame,
                          const std::vector<cv::DMatch>& inlier_matches,
                          Eigen::Isometry3f* transformation) {
  const std::size_t kNumMatches = inlier_matches.size();
  if (kNumMatches < 3) {
    return false;
  }

  Eigen::Matrix<float, 3, Eigen::Dynamic> all_inlier_from_frame_points;
  Eigen::Matrix<float, 3, Eigen::Dynamic> all_inlier_to_frame_points;
  all_inlier_from_frame_points.resize(Eigen::NoChange, kNumMatches);
  all_inlier_to_frame_points.resize(Eigen::NoChange, kNumMatches);

  for (std::size_t i = 0; i < kNumMatches; ++i) {
    all_inlier_from_frame_points.col(i) = from_frame.points[inlier_matches[i].trainIdx];
    all_inlier_to_frame_points.col(i) = to_frame.points[inlier_matches[i].queryIdx];
  }

  *transformation = Eigen::umeyama(all_inlier_from_frame_points,
                                   all_inlier_to_frame_points,
                                   false);
  return true;
};

bool GetRansacTransformation(const KeypointFrame& from_frame,
                             const KeypointFrame& to_frame,
                             const std::vector<cv::DMatch>& matches,
                             Eigen::Isometry3f* transformation) {
  // from_frame == query_points
  // to_frame == training_points
  // Umeyama gives us the inverse of what we want
  const std::vector<Eigen::Vector3f> from_frame_points = from_frame.points;
  const std::vector<Eigen::Vector3f> to_frame_points = to_frame.points;

  // Confidence threshold for early termination.
  const double kRansacConfidence = 0.9999;
  const double kLog1MinusRansacConfidence = log(1.0 - kRansacConfidence);

  // Euclidean distance threshold for inliers (10 cm).
  const double kMaxEuclDistSq = .01;

  const int kNumMatches = matches.size();
  std::size_t highest_inlier_count = 0;
  Eigen::Isometry3f best_transformation;

  // if we don't have enough matches to do RANSAC, we have to bail
  const size_t kMinSampleSize = 3;
  if (kNumMatches < kMinSampleSize) {
    return false;
  }

  for (int i = 0; i < 1000; i++) {
    int a, b, c;
    Gen3RandomIndices(kNumMatches, &a, &b, &c);

    Eigen::Matrix3f possible_from_points;
    possible_from_points.col(0) = from_frame_points[matches[a].trainIdx];
    possible_from_points.col(1) = from_frame_points[matches[b].trainIdx];
    possible_from_points.col(2) = from_frame_points[matches[c].trainIdx];
    Eigen::Matrix3f possible_to_points;
    possible_to_points.col(0) = to_frame_points[matches[a].queryIdx];
    possible_to_points.col(1) = to_frame_points[matches[b].queryIdx];
    possible_to_points.col(2) = to_frame_points[matches[c].queryIdx];
    Eigen::Isometry3f possible_transformation;
    possible_transformation =
        Eigen::umeyama(possible_from_points, possible_to_points, false);

    float score = 0;
    int inlier_count = 0;
    for (std::size_t j = 0; j < matches.size(); j++) {
      Eigen::Vector3f transformed_from_frame_point =
          TransformPoint(from_frame_points[matches[j].trainIdx],
                         possible_transformation);
      float dist_sq =
          (to_frame_points[matches[j].queryIdx] - transformed_from_frame_point).squaredNorm();
      if (dist_sq < kMaxEuclDistSq) {
        ++inlier_count;
      }
    }

    if (inlier_count > highest_inlier_count) {
      best_transformation = possible_transformation;
      highest_inlier_count = inlier_count;
    }

    // Perform confidence test to see if we can break early.
    const double best_inlier_ratio = static_cast<double>(inlier_count) /
      static_cast<double>(kNumMatches);
    const double h = kLog1MinusRansacConfidence /
      log(1.0 - pow(best_inlier_ratio, static_cast<double>(kMinSampleSize)));
    if (std::isfinite(h) && static_cast<double>(i) > h + 1.0)
        break;
  }

  // Now that we have the best fit transformation, refine transformation using all inliers.
  std::vector<cv::DMatch> inlier_matches;
  for (std::size_t j = 0; j < matches.size(); j++) {
    Eigen::Vector3f transformed_from_frame_point =
        TransformPoint(from_frame_points[matches[j].trainIdx],
                       best_transformation);
    float dist_sq =
        (to_frame_points[matches[j].queryIdx] - transformed_from_frame_point).squaredNorm();
    if (dist_sq < kMaxEuclDistSq) {
      inlier_matches.push_back(matches[j]);
    }
  }

  Eigen::Isometry3f refined_transformation;
  RefineInliersUmeyama(from_frame, to_frame, inlier_matches, &refined_transformation);

  *transformation = refined_transformation;
  return true;
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
  Mat img_depth_1_unscaled = imread(depth_1_path, CV_LOAD_IMAGE_ANYDEPTH);
  Mat img_color_2 = imread(color_2_path, CV_LOAD_IMAGE_COLOR);
  Mat img_depth_2_unscaled = imread(depth_2_path, CV_LOAD_IMAGE_ANYDEPTH);

  Mat img_depth_1, img_depth_2, img_depth_1_float, img_depth_2_float;
  img_depth_1_unscaled.convertTo(img_depth_1_float, CV_32F);
  img_depth_2_unscaled.convertTo(img_depth_2_float, CV_32F);
  img_depth_1_float.convertTo(img_depth_1, CV_32F, 1.0/10000.0);
  img_depth_2_float.convertTo(img_depth_2, CV_32F, 1.0/10000.0);

  // Get the feature points
  Ptr<Feature2D> sift = xfeatures2d::SIFT::create(300);
  vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;

  sift->detect(img_color_1, keypoints_1);
  sift->detect(img_color_2, keypoints_2);
  sift->compute(img_color_1, keypoints_1, descriptors_1);
  sift->compute(img_color_2, keypoints_2, descriptors_2);

  Mat img_keypoints1, img_keypoints2;
  drawKeypoints(img_color_1, keypoints_1, img_keypoints1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  drawKeypoints(img_color_2, keypoints_2, img_keypoints2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

  // Match up the Feature Points between images
  BFMatcher matcher;
  std::vector<std::vector<cv::DMatch>> matches;
  matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);  // Find two nearest matches
  vector<cv::DMatch> good_matches;
  for (int i = 0; i < matches.size(); ++i) {
    const float ratio = 0.6; // As in Lowe's paper; can be tuned
    if (matches[i][0].distance < ratio * matches[i][1].distance) {
      good_matches.push_back(matches[i][0]);
    }
  }

  const int max_match_distance = 10000;
  vector<DMatch> filtered_matches = filter_matches(good_matches, max_match_distance);
  //vector<DMatch> depth_filtered_matches = filter_matches_by_depth(matches, keypoints_1, keypoints_2, img_depth_1, img_depth_2, max_match_distance);
  
  vector<DMatch> depth_filtered_matches;
  for(auto match: filtered_matches) {
    Point query_keypoint = keypoints_1[match.queryIdx].pt;
    Point train_keypoint = keypoints_2[match.trainIdx].pt;

    double depth_1_value = static_cast<double>(img_depth_1.at<float>(query_keypoint.y, query_keypoint.x));
    double depth_2_value = static_cast<double>(img_depth_2.at<float>(train_keypoint.y, train_keypoint.x));
    if(depth_1_value < 1e-6 || depth_2_value < 1e-6) continue;
    cout << "After, Depth 1: " << depth_1_value << ", Depth 2: " << depth_2_value << endl;

    depth_filtered_matches.push_back(match);
  }
  
  Mat img_matches;
  drawMatches(img_color_1, keypoints_1, img_color_2, keypoints_2, depth_filtered_matches, img_matches);
  imshow("matches", img_matches);
  waitKey(0);

  // Select the matching keypoints and get their 3D representations
  Mat query_keypoints_3d, train_keypoints_3d;

  vector<Eigen::Vector3f> query_3d_eigen_points;
  vector<Eigen::Vector3f> train_3d_eigen_points;
  ReprojectKeypoints(keypoints_1, img_depth_1, camera_matrix, &query_3d_eigen_points);
  ReprojectKeypoints(keypoints_2, img_depth_2, camera_matrix, &train_3d_eigen_points);

  //rgbd::depthTo3dSparse(img_depth_1, camera_matrix, keypoints_1, query_keypoints_3d);
  //rgbd::depthTo3dSparse(img_depth_2, camera_matrix, keypoints_2, train_keypoints_3d);

  /*
  cout << endl << endl << "Filtered!!!!" << endl << endl;

  cout << "filtered_query_3d rows: " << filtered_query_3d.rows << endl;
  cout << "filtered_train_3d rows: " << filtered_train_3d.rows << endl;
  for(int i = 0; i < filtered_query_3d.rows; i++) {
    Vec3d query_vec = filtered_query_3d.at<Vec3d>(i, 0);
    Vec3d train_vec = filtered_train_3d.at<Vec3d>(i, 0);
    cout << "Query Vec: " << query_vec << ", Train Vec: " << train_vec << endl;
  }

  for(int i = 0; i < query_keypoints_3d.rows; i++) {
    Eigen::Vector3f vec(query_keypoints_3d.at<float>(i, 0), query_keypoints_3d.at<float>(i, 1), query_keypoints_3d.at<float>(i, 2));
    query_3d_eigen_points.push_back(vec);    
  }

  for(int i = 0; i < train_keypoints_3d.rows; i++) {
    Eigen::Vector3f vec(train_keypoints_3d.at<float>(i, 0), train_keypoints_3d.at<float>(i, 1), train_keypoints_3d.at<float>(i, 2));
    train_3d_eigen_points.push_back(vec);    
  }
  */

  // Print out the points
  cout << "Query 3d Eigen: " << endl;
  for(auto point: query_3d_eigen_points) {
    cout << point << endl << endl;
  }
  cout << "Train 3d Eigen: " << endl;
  for(auto point: train_3d_eigen_points) {
    cout << point << endl << endl;
  }

  KeypointFrame query_keypoint_frame, train_keypoint_frame;
  Eigen::Isometry3f transformation = Eigen::Isometry3f::Identity();

  query_keypoint_frame.rgb_image = img_color_1;
  query_keypoint_frame.depth_image = img_depth_1;
  query_keypoint_frame.keypoints = keypoints_1;
  query_keypoint_frame.descriptors = descriptors_1;
  query_keypoint_frame.points = query_3d_eigen_points;

  train_keypoint_frame.rgb_image = img_color_2;
  train_keypoint_frame.depth_image = img_depth_2;
  train_keypoint_frame.keypoints = keypoints_2;
  train_keypoint_frame.descriptors = descriptors_2;
  train_keypoint_frame.points = train_3d_eigen_points;

  bool success = GetRansacTransformation(query_keypoint_frame, train_keypoint_frame, depth_filtered_matches, &transformation);
  cout << "RansacTransformation success: " << success << endl;
  cout << "Transformation rotation: " << transformation.rotation() << endl;
  cout << "Transformation rotation matrix: " << transformation.rotation().matrix() << endl;
  cout << "Transformation translation: " << transformation.translation() << endl;

  // TODO: Estimate the affine 3d transform between images using matching keypoints
  /*
  std::vector<uchar> inliers;
  cv::Mat aff(3,4,CV_64F);
  std::cout << "Transformation: " << endl << aff << std::endl;

  // This is necesssary to calculate the inverse of the affine transformation
  Mat homogenous_row = Mat(1, 4, CV_64F);
  homogenous_row.at<double>(0, 0) = 0;
  homogenous_row.at<double>(0, 1) = 0;
  homogenous_row.at<double>(0, 2) = 0;
  homogenous_row.at<double>(0, 3) = 1;
  aff.push_back(homogenous_row);
  */

  Mat depth_1_cloud, depth_2_cloud;
  rgbd::depthTo3d(img_depth_1, camera_matrix, depth_1_cloud);
  rgbd::depthTo3d(img_depth_2, camera_matrix, depth_2_cloud);

  for(int i = 0; i < depth_2_cloud.rows; i++) {
    for(int j = 0; j < depth_2_cloud.cols; j++) {
      // Get the original Vec3d (3D point) from the Mat
      Vec3f orig_31 = depth_2_cloud.at<Vec3f>(i, j);
      Eigen::Vector3f eigen_vec3;
      eigen_vec3(0) = orig_31[0];
      eigen_vec3(1) = orig_31[1];
      eigen_vec3(2) = orig_31[2];

      // Transform the original point by the affine transform
      // 4x1 = 4x4 * 4x1
      Eigen::Vector3f result41 = transformation.inverse() * eigen_vec3;
      //cout << "result41: " << result41 << endl;

      // Convert the 4x1 Mat to a Vec3d and save it in the original mat

      Vec3f result_31(result41(0,0), result41(1,0), result41(2,0));
      depth_2_cloud.at<Vec3f>(i, j) = result_31;
      //cout << "Before: " << endl << orig_31 << endl << "After:" << endl << result_31 << endl;
    }
  }
  
  Mat depth_combined, color_combined;
  hconcat(depth_1_cloud, depth_2_cloud, depth_combined);
  hconcat(img_color_1, img_color_2, color_combined);

  viz.showWidget("cloud", viz::WCloud(depth_combined, color_combined));
  viz.spin();
  return 0;
}
