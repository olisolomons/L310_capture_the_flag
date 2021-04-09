cmake_minimum_required(VERSION 2.8.3)
project(capture_the_flag)

################################################################################
# Find catkin packages and libraries for catkin and system dependencies
################################################################################

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  message_generation
)

################################################################################
# Setup for python modules and scripts
################################################################################

catkin_python_setup()

################################################################################
# Declare catkin specific configuration to be passed to dependent projects
################################################################################

catkin_package(
  DEPENDS
    rospy
    std_msgs
    geometry_msgs
    nav_msgs
    message_runtime
)

################################################################################
# Build
################################################################################

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

################################################################################
# Install
################################################################################

catkin_install_python(PROGRAMS
  scripts/test.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY 
    scripts
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
