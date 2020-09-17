CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)
PROJECT(detnet_test)

set(CMAKE_CXX_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


OPTION(USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
   FIND_PACKAGE(OpenMP)
   IF(OPENMP_FOUND)
      SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   ENDIF()
ENDIF()

SET(AMBARELLA_PATH /home/wfw/Amba/cv25_linux_sdk_1.0/ambarella)
SET(OPENCV_LIB_PATH ${AMBARELLA_PATH}/prebuild/third-party/armv8-a/opencv/usr/lib)

# link_directories(/home/wfw/Amba/cv25_linux_sdk_1.0/ambarella/unit_test/private/dms/ncnn) 
link_directories(${AMBARELLA_PATH}/out/cv25_hazelnut/fakeroot/usr/lib/) 
link_directories(${AMBARELLA_PATH}/prebuild/third-party/armv8-a/)

#--------------set your files-------------------//
# add_definitions(-DLINUX -DNCNN_MODEL -DLANDMARK_DEBUG -DNOSHOWIMGPLAT)
include_directories(${AMBARELLA_PATH}/packages/nnctrl/inc)
include_directories(${AMBARELLA_PATH}/packages/cavalry_mem/inc)
include_directories(${AMBARELLA_PATH}/packages/vproc/inc)
include_directories(${AMBARELLA_PATH}/include/cavalry)
include_directories(${AMBARELLA_PATH}/include/arch_v5)
include_directories(${AMBARELLA_PATH}/include)
include_directories(${AMBARELLA_PATH}/packages/smartfb)
include_directories(${AMBARELLA_PATH}/boards/cv25_hazelnut)
include_directories(${AMBARELLA_PATH}/prebuild/third-party/armv8-a/opencv/include)
include_directories(${AMBARELLA_PATH}/prebuild/third-party/armv8-a/opencv/include/opencv)
include_directories(${AMBARELLA_PATH}/prebuild/third-party/armv8-a/opencv/include/opencv2)

set(OpenCV_LIBS ${OPENCV_LIB_PATH}/libopencv_imgcodecs.so.3.4
                   ${OPENCV_LIB_PATH}/libopencv_imgcodecs.so.3.4.6
                   ${OPENCV_LIB_PATH}/libopencv_calib3d.so.3.4
                   ${OPENCV_LIB_PATH}/libopencv_calib3d.so.3.4.6   
                   ${OPENCV_LIB_PATH}/libopencv_core.so.3.4
                   ${OPENCV_LIB_PATH}/libopencv_core.so.3.4.6
                    ${OPENCV_LIB_PATH}/libopencv_highgui.so.3.4
                   ${OPENCV_LIB_PATH}/libopencv_highgui.so.3.4.6    
                   ${OPENCV_LIB_PATH}/libopencv_imgproc.so.3.4
                   ${OPENCV_LIB_PATH}/libopencv_imgproc.so.3.4.6   
                   ${OPENCV_LIB_PATH}/libopencv_features2d.so.3.4
                   ${OPENCV_LIB_PATH}/libopencv_features2d.so.3.4.6   
                   ${OPENCV_LIB_PATH}/libopencv_video.so.3.4
                   ${OPENCV_LIB_PATH}/libopencv_video.so.3.4.6 
                   ${OPENCV_LIB_PATH}/libopencv_videoio.so.3.4
                   ${OPENCV_LIB_PATH}/libopencv_videoio.so.3.4.6   
                   ${OPENCV_LIB_PATH}/libopencv_flann.so.3.4
                   ${OPENCV_LIB_PATH}/libopencv_flann.so.3.4.6
                  )

set(Other_LIBS ${AMBARELLA_PATH}/prebuild/third-party/armv8-a/libjpeg-turbo/usr/lib/libjpeg.so.8
                  ${AMBARELLA_PATH}/prebuild/third-party/armv8-a/libpng/usr/lib/libpng.so
                  ${AMBARELLA_PATH}/prebuild/third-party/armv8-a/libpng/usr/lib/libpng16.so
                  ${AMBARELLA_PATH}/prebuild/third-party/armv8-a//libpng/usr/lib/libpng16.so.16
                  ${AMBARELLA_PATH}/prebuild/third-party/armv8-a/libpng/usr/lib/libpng16.so.16.27.0
                  ${AMBARELLA_PATH}/prebuild/third-party/armv8-a/zlib/usr/lib/libz.so.1
                  ${AMBARELLA_PATH}/out/cv25_hazelnut/packages/nnctrl/libnnctrl.so
                  ${AMBARELLA_PATH}/out/cv25_hazelnut/packages/vproc/libvproc.so
                  ${AMBARELLA_PATH}/out/cv25_hazelnut/packages/cavalry_mem/libcavalry_mem.so
                  ${AMBARELLA_PATH}/out/cv25_hazelnut/packages/smartfb/libsmartfb.so
                  ${AMBARELLA_PATH}/prebuild/third-party/armv8-a/freetype/usr/lib/libfreetype.so
                  )

#---------------------------include----------------------------#
include_directories(${CMAKE_SOURCE_DIR}/inc)

#-----------------------------source----------------------------#
FILE(GLOB_RECURSE SRC_LIST  
          ./detnet_test.cpp
          ./src/utils.cpp
          ./src/yolov3.cpp
          )

#set(caffe_dep_lib caffe glog boost_filesystem boost_system)

add_executable(detnet_test ./detnet_test.cpp  ${SRC_LIST})
target_link_libraries(detnet_test ${OpenCV_LIBS} ${Other_LIBS})

# To compile this program all you need to do is ask cmake.  You would type
# these commands from within the directory containing this CMakeLists.txt
# file:
#   mkdir build
#   cd build
#   cmake ..
#   cmake --build . --config Release
