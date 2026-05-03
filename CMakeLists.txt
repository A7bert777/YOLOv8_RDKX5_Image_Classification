cmake_minimum_required(VERSION 3.10)

project(rdkx5_yolov8_cls)

# C++11 standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
# libdnn.so depends on system software dynamic link library
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,-unresolved-symbols=ignore-in-shared-libs")
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# D-Robotics BPU DNN library
# CMAKE_CURRENT_SOURCE_DIR 会自动获取当前 CMakeLists.txt 所在的绝对路径
set(DNN_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(DNN_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/rdk_lib")

include_directories(${DNN_PATH})
link_directories(${DNN_LIB_PATH})

# Add executable
add_executable(rdkx5_yolov8_cls src/main.cc)

# ==========================================
# 4. 链接库与 RPATH
# ==========================================
target_link_libraries(rdkx5_yolov8_cls
    ${OpenCV_LIBS}
    dnn       
    cnn_intf  
    hbmem     
    hbrt_bayes_aarch64
    alog      
    pthread
    rt
    dl
)



# 强制链接器禁用 new-dtags，从而生成 RPATH 而非 RUNPATH
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")

set_target_properties(rdkx5_yolov8_cls PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    # 意思是：运行期去 "当前可执行文件所在目录的上一级的 rdk_lib" 里找库
    INSTALL_RPATH "\$ORIGIN/../rdk_lib" 
)