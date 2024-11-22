// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <stdexcept>
#include <frame.h>
#include <feature.h>
#include <point.h>
#include <boost/bind.hpp>
#include <vikit/math_utils.h>
#include <vikit/vision.h>
#include <vikit/performance_monitor.h>
// #include <fast/fast.h>

namespace lidar_selection { // this namespace is used to store relative information about lidar input.

int Frame::frame_counter_ = 0; // initialize the frame_counter_ to be 0 at the first.

// constructor, initialize id_ member to be global unique frame_counter_ value, and cam_ member to be cam, key_pts_ member to be 5, is_keyframe_ to be false.(defaultly)
// use initFrame to implement the detail of constructor.
Frame::Frame(vk::AbstractCamera* cam, const cv::Mat& img) :
    id_(frame_counter_++), 
    cam_(cam), 
    key_pts_(5), 
    is_keyframe_(false)
{
  initFrame(img);
}

// reset features
Frame::~Frame()
{
  std::for_each(fts_.begin(), fts_.end(), [&](FeaturePtr i){i.reset();});
}

// initialize new frame. Implementation details of a constructor.
void Frame::initFrame(const cv::Mat& img)
{
  // check image
  if(img.empty() || img.type() != CV_8UC1 || img.cols != cam_->width() || img.rows != cam_->height())
    throw std::runtime_error("Frame: provided image has not the same size as the camera model or image is not grayscale");

  // Set keypoints to nullptr
  std::for_each(key_pts_.begin(), key_pts_.end(), [&](FeaturePtr ftr){ ftr=nullptr; });
  
  ImgPyr ().swap(img_pyr_);
  img_pyr_.push_back(img);
  // Build Image Pyramid
  // frame_utils::createImgPyramid(img, max(Config::nPyrLevels(), Config::kltMaxLevel()+1), img_pyr_);
  // frame_utils::createImgPyramid(img, 5, img_pyr_); 
}

void Frame::setKeyframe()
{
  is_keyframe_ = true;
  // we need to set key points for those keyframes. KeyPoints can be used to check overlapping FoV
  setKeyPoints();
}

void Frame::addFeature(FeaturePtr ftr)
{
  fts_.push_back(ftr);
}

// implementation of setKeyPoints
void Frame::setKeyPoints()
{
  // check for the five key_pts_ feature, if the key_pts_ feature don't have a 3D points bind to it, then set this feature in key_pts_ to be nullptr
  for(size_t i = 0; i < 5; ++i)
    if(key_pts_[i] != nullptr)
      if(key_pts_[i]->point == nullptr) // feature's corresponding 3D point
        key_pts_[i] = nullptr;
  // traverse the fts_ features in the image. For every feature with a point, checkKeyPoints. 
  // In this function, we will fulfill the key_pts_ with best key points to do overlapping fov check
  std::for_each(fts_.begin(), fts_.end(), [&](FeaturePtr ftr){ if(ftr->point != nullptr) checkKeyPoints(ftr); });
}

// part of implementation for setKeyPoints
// after the checking traverse, the best fitting feature of center and 4 corners will be matched.
// TODO: the five points checking can be parallized using GPU or multithread. (but the management cost might be higher than the time sa)
void Frame::checkKeyPoints(FeaturePtr ftr)
{
  const int cu = cam_->width()/2; // calculate the central value of u(width)
  const int cv = cam_->height()/2; // calculate the central value of v(height)

  // center pixel
  if(key_pts_[0] == nullptr) // first initialize this point to the checking feature.
    key_pts_[0] = ftr;
  else if(std::max(std::fabs(ftr->px[0]-cu), std::fabs(ftr->px[1]-cv)) // feature's corresponding 2d coordinates in pyramid level 0(original image)
        < std::max(std::fabs(key_pts_[0]->px[0]-cu), std::fabs(key_pts_[0]->px[1]-cv)))
    key_pts_[0] = ftr; // if checking feature is closer to the central point, refresh key_pts_[0](center pixel) to be a new one.
  
  // top right corner refreshing
  if(ftr->px[0] >= cu && ftr->px[1] >= cv)
  {
    if(key_pts_[1] == nullptr)
      key_pts_[1] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          > (key_pts_[1]->px[0]-cu) * (key_pts_[1]->px[1]-cv))
      key_pts_[1] = ftr;
  }

  // bottom right corner refreshing
  if(ftr->px[0] >= cu && ftr->px[1] < cv)
  {
    if(key_pts_[2] == nullptr)
      key_pts_[2] = ftr;
    // else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
    else if((ftr->px[0]-cu) * (cv-ftr->px[1])
          // > (key_pts_[2]->px[0]-cu) * (key_pts_[2]->px[1]-cv))
          > (key_pts_[2]->px[0]-cu) * (cv-key_pts_[2]->px[1]))
      key_pts_[2] = ftr;
  }

  // bottom left corner refreshing
  if(ftr->px[0] < cu && ftr->px[1] < cv)
  {
    if(key_pts_[3] == nullptr)
      key_pts_[3] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          > (key_pts_[3]->px[0]-cu) * (key_pts_[3]->px[1]-cv))
      key_pts_[3] = ftr;
  }

  // top left corner refreshing
  if(ftr->px[0] < cu && ftr->px[1] >= cv)  
  // if(ftr->px[0] < cv && ftr->px[1] >= cv)
  {
    if(key_pts_[4] == nullptr)
      key_pts_[4] = ftr;

    else if(cu-(ftr->px[0]) * (ftr->px[1]-cv) 
          > (cu-key_pts_[4]->px[0]) * (key_pts_[4]->px[1]-cv))      
      key_pts_[4] = ftr;
  }
}

// implementation of keypoint remove when a point/feature is deleted.
void Frame::removeKeyPoint(FeaturePtr ftr)
{
  bool found = false;
  std::for_each(key_pts_.begin(), key_pts_.end(), [&](FeaturePtr& i){
    if(i == ftr) {
      i = nullptr;
      found = true;
    }
  });
  if(found) // find a new key point as a substitute, call setKeyPoints again.
    setKeyPoints();
}

// check whether world coordinate's point is inside the frustum of camera coordinate
bool Frame::isVisible(const Vector3d& xyz_w) const
{
  Vector3d xyz_f = T_f_w_*xyz_w; // transform the world coordinate to frame coordinate (SE3 * Vector3d -> Vector3d)

  if(xyz_f.z() < 0.0) // can not be viewed because of the z value is less than camera 
    return false; // point is behind the camera
  Vector2d px = f2c(xyz_f); // f2c is a function to transform frame 3d coordinate to 2d uv coordinate

  if(px[0] >= 0.0 && px[1] >= 0.0 && px[0] < cam_->width() && px[1] < cam_->height())
    return true; // in the scope of the camera, return true.
  return false;
}

/// Utility functions for the Frame class
namespace frame_utils {

void createImgPyramid(const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr)
{
  pyr.resize(n_levels);
  pyr[0] = img_level_0;

  for(int i=1; i<n_levels; ++i)
  {
    pyr[i] = cv::Mat(pyr[i-1].rows/2, pyr[i-1].cols/2, CV_8U);
    vk::halfSample(pyr[i-1], pyr[i]); // half sampling the image as a pyramid
  }
}


bool getSceneDepth(const Frame& frame, double& depth_mean, double& depth_min)
{
  vector<double> depth_vec;
  depth_vec.reserve(frame.fts_.size()); // space reserved for frame feature number
  depth_min = std::numeric_limits<double>::max(); // initialized to be positive infinate for double
  for(auto it=frame.fts_.begin(), ite=frame.fts_.end(); it!=ite; ++it) // iterate over all features in current frame.
  {
    if((*it)->point != nullptr) 
    {
      const double z = frame.w2f((*it)->point->pos_).z();
      depth_vec.push_back(z);
      depth_min = fmin(z, depth_min);
    }
  }
  if(depth_vec.empty())
  {
    cout<<"Cannot set scene depth. Frame has no point-observations!"<<endl;
    return false;
  }
  depth_mean = vk::getMedian(depth_vec);
  return true;
}

} // namespace frame_utils
} // namespace lidar_selection
