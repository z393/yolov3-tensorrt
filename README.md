## YOLOV3
本项目将旨在说明如何将在pytorch的框架下训练yolov3,并且将yolov3的.pt的pytorch模型、caffe模型以及darknet模型进行相互的转化，以及如何使用tensorRT的工程将caffe的模型向tensorRT的engine模型进行转化，更进一步的，将会介绍从caffe模型转换成为xilinx的FPGA可以加速的模型。

### yolov3-pytorch
This [yolov3-pytorch](./yolov3-pytorch/README.md) contains PyTorch YOLOv3 software developed by Ultralytics LLC, and is freely available for redistribution under the GPL-3.0 license. For more information please visit https://www.ultralytics.com.  

#### 在此处添加你的Training文档
训练代码：  


官方模型文件：  
`/home/hy/ethan/yolov3/yolov3-pytorch/cfg/official.cfg`  
`/home/hy/ethan/yolov3/yolov3-pytorch/weights/official/weights`  

训练模型文件：  
`/home/hy/z393/yolov3/yolov3-pytorch/cfg/high-speed-yolov3-20191030.cfg`  
`/home/hy/z393/yolov3/yolov3-pytorch/weights/high-speed-yolov3-20191030.weights`  

从Val文件目录中随机生成一定数量的文件作为val目录：  
`python gen_val_samples.py --data data/high-speed-yolov3-20191030.data --output data/val-samples --num 30`  
对应的测试文件目录：  
`/home/hy/z393/yolov3/yolov3-pytorch/data/val-samples`  

执行以下命令生成对应的bbox渲染图：（官方原始YOLOV3）  
`python detect.py --source data/val-samples/ --cfg cfg/official.cfg --weight weights/official.weights --data data/official.data`  

执行以下命令生成对应的bbox渲染图：（训练生成YOLOV3）  
`python detect.py --source data/val-samples/ --cfg cfg/high-speed-yolov3-20191030.cfg --weight weights/high-speed-yolov3-20191030.weights --data data/high-speed-yolov3-20191030.data`  

---

#### 在此处添加你的Inference文档

#### 模型转化
此处我们以yolov3的官方文件以及我们在该工程下通过我们自己的数据集训练的pt模型进行转换的测试，主要进行的是将pytorch的模型转化成为yolo darket的模型。  
1. 下载训练好的模型的配置文件[huanyu_high_speed_yolov3_20191030](http://47.100.39.180/download/inDriving/model/yolo/yolov3-pytorch/cfg/high-speed-yolov3-20191030.cfg)  
在此处，我们是在原有的yolov3.cfg的基础上进行的重新训练，因此模型配置文件`high-speed-yolov3-20191030.cfg`与官方的`yolov3.cfg`的文件是一样的。  
2. 下载训练好的权重文件[huanyu_high_speed_yolov3_20191030](http://47.100.39.180/download/inDriving/model/yolo/yolov3-pytorch/weights/high-speed-yolov3-20191030.pt)  
由于是在pytorch框架下训练的，因此训练出的模型是.pt的格式  
3. 将上述文件拷贝到相应的目录下  
`cp ./high-speed-yolov3-20191030.cfg ./cfg/high-speed-yolov3-20191030.cfg`  
`cp ./high-speed-yolov3-20191030.pt ./weights/high-speed-yolov3-20191030.pt`  
4. 转换模型  
由于该工程需要依赖torch等一些环境，因此系统根据需求创建了一个anacoda的环境`yolov3`，具体需要的环境，可以查看[yolov3-pytorch](./yolov3-pytorch/README.md)帮助文档  
`source ~/anaconda3/bin/activate yolov3`  
`python  -c "from models import *; convert('cfg/high-speed-yolov3-20191030.cfg', 'weights/high-speed-yolov3-20191030.pt')"`  
正常情况下可以看到这样的打印`Success: converted 'weights/high-speed-yolov3-20191030.pt' to 'converted.weights'`,在当前目录下出现了`converted.weights`  
`mv converted.weights ./weights/high-speed-yolov3-20191030.weights`  

至此，**.pt**的pytorch工程的模型文件就转成了**darknet**的模型文件了，分别保存在`cfg/high-speed-yolov3-20191030.cfg`以及`./weights/high-speed-yolov3-20191030.weights`中
---  

### darknet2caffe
使用该工程的脚本可以将darknet的模型转换成为caffe的模型，值得注意的是，这个工程是python2的，因此需要你的caffe环境是在python2下面编译的  
1. 建立你的环境  
`source ~/anaconda3/bin/activate yolo_convertor`  
在该anaconda环境下，我们安装了Python2以及torch环境，并且caffe是在python2下面安装的  
`echo $PYTHONPATH`  
    ```
    /home/hy/z393/titan_dataset_studio/tools/catkin_ws/devel/lib/python2.7/dist-packages:/opt/ros/kinetic/lib/python2.7/dist-packages:/home/hy/ethan/caffe-ssd-python2/python
    ```
2. 将之前生成的darknet的模型复制到对应目录  
具体而言将`.cfg`文件复制到`./cfg`目录下，将`.weights`文件复制到`./weights`目录下。  
3. 模型转换  
`python darknet2caffe.py cfg/high-speed-yolov3-20191030.cfg weights/high-speed-yolov3-20191030.weights prototxt/high-speed-yolov3-20191030.prototxt caffemodel/high-speed-yolov3-20191030.caffemodel`  
模型转换成功后，你能够在`./prototxt`里面发现`high-speed-yolov3-20191030.prototxt`以及在`./caffemodel`里面发现`high-speed-yolov3-20191030.caffemodel`  

---

### TensorRT-Yolov3
使用该工程的脚本可以将caffe的模型转换成为tensorRT加速的`.engine`的模型，并且附带了一些测试案例  
1. 编译你的环境  
`mkdir build`  
`cd build`  
`cmake ..`  
`make & make install`  
`cd ..`  
2. 编译你的yolo-608模型  
因为不同的输入分辨率对应不同的YOLO-KERNEL的配置，所以我们在代码中通过宏来定义的，如果你需要编译出608的模型，修改CMakeLists.txt  
添加一行:  
`add_definitions(-DYOLO608) # or define YOLO416 for yolo416 model`  
重新执行第一步编译步骤  
3. 准备你的CAFFE模型  
具体而言将`.prototxt`文件复制到`./prototxt`目录下，将`.caffemodel`文件复制到`./caffemodel`目录下。  
4. 修改你的配置文件  
正常情况下不修改配置文件`.prototxt`的情况下会出现错误:  <font color=red>```[libprotobuf ERROR google/protobuf/text_format.cc:298] Error parsing text-format ditcaffe.NetParameter: 2622:20: Message type "ditcaffe.LayerParameter" has no field named "upsample_param"```</font>  
该错误是由于默认的proto文件不能支持这个上采样的参数，直接注释掉就可以：  
<font color=blue>```Hi，@Ricardozzf. TensorRT caffe parser can't check the param not in it's default proto file, although I added it as plugin. You need to comment the "upsample_param" but still leave the type "Upsample". Then the running can be OK.```</font>  
修改`./prototxt/high-speed-yolov3-20191030.prototxt`文件，注释掉：  
    ```
    #upsample_param {
    #    scale: 2
    #}
    ```
    除此以外，还需要修改的是在最后一层增加一个yolo_det的输出，由于默认的caffe里面没有yolo_det的实现，而我们在这个tensorRT里有yoloLayer.cu的实现，我们需要配置文件里面有yolo_det层。  
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
