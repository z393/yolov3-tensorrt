## 所需环境
python2.7

caffe

## caffe中加入upsample layer层
1. 把upsample_layer.hpp 放在include/caffe/layers下面；

2. 把upsample_layer.cpp与upsample_layer.cu放在src/caffe/layers下面；

3. 修改相应的caffe.proto文件，src/caffe/proto/caffe.proto中的LayerParameter的最后一行加入加入：

       　message LayerParameter {
         .....
         optional UpsampleParameter upsample_param = 149;
         }
  
4. 再caffe.proto中添加upsample层的参数
       
        message UpsampleParameter{
        optional int32 scale = 1 [default = 1];
        }
        
5. 重新编译caffe

## 模型转换
运行darknet2caffe.py将.weights模型转换得到caffemodel和prototxt

        python darknet2caffe.py cfg/high-speed-yolov3-20191030.cfg weights/high-speed-yolov3-20191030.weights prototxt/high-speed-yolov3-20191030.prototxt caffemodel/high-speed-yolov3-20191030.caffemodel
