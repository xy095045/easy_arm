SET(AMBARELLA_DIR /home/lpj/Software/cv22_linux_sdk/ambarella)

#fakeroot
link_directories(${AMBARELLA_DIR}/out/cv22_walnut/fakeroot/usr/lib)

#amba bsp
include_directories(${AMBARELLA_DIR}/include)
include_directories(${AMBARELLA_DIR}/include/arch_v5)
include_directories(${AMBARELLA_PATH}/boards/cv22_walnut)

#nnctrl
include_directories(${AMBARELLA_DIR}/packages/nnctrl/inc)
link_directories(${AMBARELLA_DIR}/out/cv22_walnut/packages/nnctrl)

#cavalry
include_directories(${AMBARELLA_DIR}/include/cavalry)
link_directories(${AMBARELLA_DIR}/out/cv22_walnut/packages/nnctrl)
include_directories(${AMBARELLA_DIR}/packages/cavalry_mem/inc)
link_directories(${AMBARELLA_DIR}/out/cv22_walnut/packages/cavalry_mem)

#smartfb
include_directories(${AMBARELLA_DIR}/packages/smartfb)
link_directories(${AMBARELLA_DIR}/out/cv22_walnut/packages/smartfb)

#vproc
include_directories(${AMBARELLA_DIR}/packages/vproc/inc)
link_directories(${AMBARELLA_DIR}/out/cv22_walnut/packages/vproc)

#third-party
link_directories(${AMBARELLA_DIR}/prebuild/third-party/armv8-a/libjpeg-turbo/usr/lib)
link_directories(${AMBARELLA_DIR}/prebuild/third-party/armv8-a/libpng/usr/lib)
link_directories(${AMBARELLA_DIR}/prebuild/third-party/armv8-a/zlib/usr/lib)
link_directories(${AMBARELLA_DIR}/prebuild/third-party/armv8-a/freetype/usr/lib)

#opencv
set(OpenCV_DIR ${AMBARELLA_DIR}/prebuild/third-party/armv8-a/opencv/usr/lib/OpenCV)
find_package(OpenCV REQUIRED CONFIG)
message(STATUS "Found OpenCV ${OpenCV_VERSION}")
message(STATUS "${OpenCV_INCLUDE_DIRS}, ${OpenCV_LIBS}")
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBS})

OPTION(USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
   FIND_PACKAGE(OpenMP)
   IF(OPENMP_FOUND)
      SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   ENDIF()
ENDIF()