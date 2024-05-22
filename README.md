# CenterNet_rknn_Cplusplus
centernet 瑞芯微 rknn 板端 C++部署。

## 编译和运行

1）编译

```
cd examples/rknn_rknn_centernet_demo_open

bash build-linux_RK3588.sh

```

2）运行

```
cd install/rknn_centernet_demo_Linux

./rknn_centernet_demo

```

注意：修改模型、测试图像、保存图像的路径，修改文件为src下的main.cc

```

int main(int argc, char **argv)
{
    char model_path[256] = "/home/zhangqian/rknn/examples/rknn_centernet/model/RK3588/yolov8nseg_relu_80class_dfl.rknn";
    char image_path[256] = "/home/zhangqian/rknn/examples/rknn_centernet/test.png";
    char save_image_path[256] = "/home/zhangqian/rknn/examples/rknn_centernet/test_result.jpg";

    detect(model_path, image_path, save_image_path);
    return 0;
}
```


# 板端测试效果

冒号“:”前的数子是coco的80类对应的类别，后面的浮点数是目标得分。（类别:得分）

![images](https://github.com/cqu20160901/CenterNet_rknn_Cplusplus/blob/main/examples/rknn_centernet/test_result.jpg)




