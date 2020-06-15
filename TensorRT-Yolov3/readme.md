
### 测试环境

    Ubuntu  16.04
    TensorRT 4.0.1.6
    CUDA 9.0
    CUDNN 7.3.1
    opencv 3.3.1
    ros kinetic

### TensorRT-Yolov3
使用该工程的脚本可以将caffe的模型转换成为tensorRT加速的`.engine`的模型，并且附带了一些测试案例  
1. 编译环境  
`mkdir build`  
`cd build`  
`cmake ..`  
`make & make install`  
`cd ..`  
2. 编译你的yolo-608模型  
因为不同的输入分辨率对应不同的YOLO-KERNEL的配置，所以在代码中通过宏来定义的，如果需要编译出608的模型，修改CMakeLists.txt  
添加一行:  
`add_definitions(-DYOLO608) # or define YOLO416 for yolo416 model`  
重新执行第一步编译步骤  
3. 准备CAFFE模型  
具体而言将`.prototxt`文件复制到`./prototxt`目录下，将`.caffemodel`文件复制到`./caffemodel`目录下。  
4. 修改配置文件  
正常情况下不修改配置文件`.prototxt`的情况下会出现错误:  <font color=red>```[libprotobuf ERROR google/protobuf/text_format.cc:298] Error parsing text-format ditcaffe.NetParameter: 2622:20: Message type "ditcaffe.LayerParameter" has no field named "upsample_param"```</font>  
该错误是由于默认的proto文件不能支持这个上采样的参数，直接注释掉就可以：  
<font color=blue>```Hi，@Ricardozzf. TensorRT caffe parser can't check the param not in it's default proto file, although I added it as plugin. You need to comment the "upsample_param" but still leave the type "Upsample". Then the running can be OK.```</font>  
修改`./prototxt/high-speed-yolov3-20191030.prototxt`文件，注释掉：  
    ```
    #upsample_param {
    #    scale: 2
    #}
    ```
    除此以外，还需要修改的是在最后一层增加一个yolo_det的输出，由于默认的caffe里面没有yolo_det的实现，而在这个tensorRT里有yoloLayer.cu的实现，我们需要配置文件里面有yolo_det层。  
    修改`./prototxt/high-speed-yolov3-20191030.prototxt`文件，在最后增加：  
    ```
    layer {
        bottom: "layer82-conv"
        bottom: "layer94-conv"
        bottom: "layer106-conv"
        top: "yolo-det"
        name: "yolo-det"
        type: "Yolo"
    }
    ```
5. 进行tensorRT模型转换  
`./install/convert_engine --caffemodel=./caffemodel/high-speed-yolov3-20191030.caffemodel --prototxt=./prototxt/high-speed-yolov3-20191030.prototxt --W=608 --H=608 --class=80 --mode=fp16`  
如上是将608x608的模型转换成tensorRT的FP16精度的engine,转换成功后会生成一个`yolov3_fp16.engine`的文件。  
<font color=red>值得注意的是tensorRT的模型转换是依赖平台的，也就是在x86机器上转换的tensorRT在NVIDIA的嵌入式处理器上是不能使用的，因此最后需要在哪个平台上运行，就在哪个平台上转换</font>  
6. 运行一个转换模型并且输出探测结果的例子  
`./install/main_test --caffemodel=./caffemodel/high-speed-yolov3-20191030.caffemodel --prototxt=./prototxt/high-speed-yolov3-20191030.prototxt --W=608 --H=608 --class=80 --input=./test/000156.png`  
<div align=center><img src="img/tensorRT_det.png"/></div>  


### Performance

Model | GPU | Mode | Inference Time
-- | -- | -- | -- 
Yolov3-416 |  GTX 1060 | Caffe | 54.593ms
Yolov3-416 |  GTX 1060 | float32 | 23.817ms
Yolov3-416 |  GTX 1060 | int8 | 11.921ms
Yolov3-608 |  GTX 1060 | Caffe | 88.489ms
Yolov3-608 | GTX 1060 | float32 | 43.965ms
Yolov3-608 |  GTX 1060 | int8 | 21.638ms
Yolov3-608 | GTX 1080 Ti | float32 | 19.353ms
Yolov3-608 | GTX 1080 Ti | int8 | 9.727ms
Yolov3-416 |  GTX 1080 Ti | float32 | 9.677ms
Yolov3-416 |  GTX 1080 Ti | int8 | 6.129ms  | li


### 参考
https://github.com/lewes6369/TensorRT-Yolov3
