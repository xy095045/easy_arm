PROJECT(detection2d)

#include amba_arm directories
include_directories(${amba_arm_BINARY_DIR})
include_directories(${amba_arm_SOURCE_DIR})

#collect source files
file(GLOB COMMON_SRCS
    "${amba_arm_SOURCE_DIR}/inference/common/*.cpp"
)

add_executable(detnet_test classnet_test.cpp yolo.cpp ${COMMON_SRCS})
target_link_libraries(detnet_test 
                      ${OpenCV_LIBS} 
                      jpeg
                      png
                      z
                      nnctrl
                      vproc
                      cavalry_mem)