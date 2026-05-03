# YOLOv8_RDKX5_Image_Classification

CSDN地址：[【YOLOv8cls部署至RDK X5】模型训练→转换bin→Sunrise 5部署](https://blog.csdn.net/A_l_b_ert/article/details/160743483?spm=1001.2014.3001.5502)

QQ咨询（not free）：2506245294

# 目标检测仓库

切记：一定要在RDK系列开发板上运行，不要在虚拟机上跑，ARM64和X86不一样！

1.项目代码介绍

src/main.cc ：程序运行文件

include/* ：各函数声明

2.配置文件介绍

build 是编译位置

inputimage 是输入图片所在文件夹

outputimage 是输出图片所在文件夹

model 是.bin模型所在文件夹

rdk_lib 是地瓜官方动态库libdnn.so等文件所在位置

3.编译运行

先删除build文件夹所有内容

**①cd build**

**②cmake ..**

**③make**

**④./rdkx5_yolov8_cls**

此处统一说明：加QQ后直接说问题和小星星截图，对于常见的相同问题，很多都已在CSDN博客中提到了（RDK的转换流程是统一的，可去博主所有的RKNN相关博客下去翻评论），已在评论中详细解释过的问题，不予回复。
