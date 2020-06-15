#include <opencv2/opencv.hpp>
#include "TrtNet.h"
#include "argsParser.h"
#include "configs.h"
#include <chrono>
#include "YoloLayer.h"
#include "dataReader.h"
#include "eval.h"

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

using namespace std;
using namespace argsParser;
using namespace Tn;
using namespace Yolo;

std::unique_ptr<trtNet> net;
int batchSize;
string engineName;
int outputCount;
unique_ptr<float[]> outputData;
list<vector<Bbox>> outputs;
int classNum;
int c;
int h;
int w;
int batchCount = 0;
vector<float> inputData;

ros::Publisher yolo_trt_detection_result_;
image_transport::Subscriber image_sub_;

vector<float> prepareImage(cv::Mat& img)
{
    using namespace cv;

    int c = parser::getIntValue("C");
    int h = parser::getIntValue("H");   //net h
    int w = parser::getIntValue("W");   //net w

    float scale = min(float(w)/img.cols,float(h)/img.rows);
    auto scaleSize = cv::Size(img.cols * scale,img.rows * scale);

    cv::Mat rgb ;
    cv::cvtColor(img, rgb, CV_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized,scaleSize,0,0,INTER_CUBIC);

    cv::Mat cropped(h, w,CV_8UC3, 127);
    Rect rect((w- scaleSize.width)/2, (h-scaleSize.height)/2, scaleSize.width,scaleSize.height); 
    resized.copyTo(cropped(rect));

    cv::Mat img_float;
    if (c == 3)
        cropped.convertTo(img_float, CV_32FC3, 1/255.0);
    else
        cropped.convertTo(img_float, CV_32FC1 ,1/255.0);

    //HWC TO CHW
    vector<Mat> input_channels(c);
    cv::split(img_float, input_channels);

    vector<float> result(h*w*c);
    auto data = result.data();
    int channelLength = h * w;
    for (int i = 0; i < c; ++i) {
        memcpy(data,input_channels[i].data,channelLength*sizeof(float));
        data += channelLength;
    }

    return result;
}

void DoNms(vector<Detection>& detections,int classes ,float nmsThresh)
{
    auto t_start = chrono::high_resolution_clock::now();

    vector<vector<Detection>> resClass;
    resClass.resize(classes);

    for (const auto& item : detections)
        resClass[item.classId].push_back(item);

    auto iouCompute = [](float * lbox, float* rbox)
    {
        float interBox[] = {
            max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
            min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
            max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
            min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
        };
        
        if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
            return 0.0f;

        float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
        return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
    };

    vector<Detection> result;
    for (int i = 0;i<classes;++i)
    {
        auto& dets =resClass[i]; 
        if(dets.size() == 0)
            continue;

        sort(dets.begin(),dets.end(),[=](const Detection& left,const Detection& right){
            return left.prob > right.prob;
        });

        for (unsigned int m = 0;m < dets.size() ; ++m)
        {
            auto& item = dets[m];
            result.push_back(item);
            for(unsigned int n = m + 1;n < dets.size() ; ++n)
            {
                if (iouCompute(item.bbox,dets[n].bbox) > nmsThresh)
                {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }

    //swap(detections,result);
    detections = move(result);

    auto t_end = chrono::high_resolution_clock::now();
    float total = chrono::duration<float, milli>(t_end - t_start).count();
    cout << "Time taken for nms is " << total << " ms." << endl;
}


vector<Bbox> postProcessImg(cv::Mat& img,vector<Detection>& detections,int classes)
{
    using namespace cv;

    int h = parser::getIntValue("H");   //net h
    int w = parser::getIntValue("W");   //net w

    //scale bbox to img
    int width = img.cols;
    int height = img.rows;
    float scale = min(float(w)/width,float(h)/height);
    float scaleSize[] = {width * scale,height * scale};

    //correct box
    for (auto& item : detections)
    {
        auto& bbox = item.bbox;
        bbox[0] = (bbox[0] * w - (w - scaleSize[0])/2.f) / scaleSize[0];
        bbox[1] = (bbox[1] * h - (h - scaleSize[1])/2.f) / scaleSize[1];
        bbox[2] /= scaleSize[0];
        bbox[3] /= scaleSize[1];
    }
    
    //nms
    float nmsThresh = parser::getFloatValue("nms");
    if(nmsThresh > 0) 
        DoNms(detections,classes,nmsThresh);

    vector<Bbox> boxes;
    for(const auto& item : detections)
    {
        auto& b = item.bbox;
        Bbox bbox = 
        { 
            item.classId,   //classId
            max(int((b[0]-b[2]/2.)*width),0), //left
            min(int((b[0]+b[2]/2.)*width),width), //right
            max(int((b[1]-b[3]/2.)*height),0), //top
            min(int((b[1]+b[3]/2.)*height),height), //bot
            item.prob       //score
        };
        boxes.push_back(bbox);
    }

    return boxes;
}

vector<string> split(const string& str, char delim)
{
    stringstream ss(str);
    string token;
    vector<string> container;
    while (getline(ss, token, delim)) {
        container.push_back(token);
    }

    return container;
}

void onImageCallback(const sensor_msgs::Image::ConstPtr& msg){
	cv_bridge::CvImagePtr cv_ptr;
	try
	{
		cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}
	
	cv::Mat img = cv_ptr->image;
	
	vector<float> curInput = prepareImage(img);
	if (!curInput.data())
		return;

	inputData.reserve(h*w*c);
	
	inputData.insert(inputData.end(), curInput.begin(), curInput.end());
	
	net->doInference(inputData.data(), outputData.get(),1);
	
	//Get Output    
	auto output = outputData.get();
	auto outputSize = net->getOutputSize()/ sizeof(float);

	//first detect count
	int detCount = output[0];
	//later detect result
	vector<Detection> result;
	result.resize(detCount);
	memcpy(result.data(), &output[1], detCount*sizeof(Detection));

	auto boxes = postProcessImg(img,result,classNum);
	outputs.emplace_back(boxes);

	output += outputSize;
	
	inputData.clear();
	
	net->printTime(); 
	
	auto bbox = *outputs.begin();
	for(const auto& item : bbox)
	{
		cv::rectangle(img,cv::Point(item.left,item.top),cv::Point(item.right,item.bot),cv::Scalar(0,0,255),3,8,0);
	}
	
	outputs.clear();
	
	cv_bridge::CvImage cv_image;
	cv_image.image = img;
	cv_image.encoding = "bgr8";
	sensor_msgs::Image ros_image;
	cv_image.toImageMsg(ros_image);
	yolo_trt_detection_result_.publish(ros_image);
}


int main( int argc, char* argv[] )
{
    // declear ros
    ros::init(argc, argv, "yolov3_trt_ros");
	ros::NodeHandle nh_;
	
	image_transport::ImageTransport it(nh_);
	yolo_trt_detection_result_ = nh_.advertise<sensor_msgs::Image>("yolo_trt_detection_result", 1);
	image_sub_ = it.subscribe("/cam_front/csi_cam/image_raw", 1, onImageCallback);
	
    parser::ADD_ARG_STRING("prototxt",Desc("input yolov3 deploy"),DefaultValue(INPUT_PROTOTXT),ValueDesc("file"));
    parser::ADD_ARG_STRING("caffemodel",Desc("input yolov3 caffemodel"),DefaultValue(INPUT_CAFFEMODEL),ValueDesc("file"));
    parser::ADD_ARG_INT("C",Desc("channel"),DefaultValue(to_string(INPUT_CHANNEL)));
    parser::ADD_ARG_INT("H",Desc("height"),DefaultValue(to_string(INPUT_HEIGHT)));
    parser::ADD_ARG_INT("W",Desc("width"),DefaultValue(to_string(INPUT_WIDTH)));
    parser::ADD_ARG_STRING("calib",Desc("calibration image List"),DefaultValue(CALIBRATION_LIST),ValueDesc("file"));
    parser::ADD_ARG_STRING("mode",Desc("runtime mode"),DefaultValue(MODE), ValueDesc("fp32/fp16/int8"));
    parser::ADD_ARG_STRING("outputs",Desc("output nodes name"),DefaultValue(OUTPUTS));
    parser::ADD_ARG_INT("class",Desc("num of classes"),DefaultValue(to_string(DETECT_CLASSES)));
    parser::ADD_ARG_FLOAT("nms",Desc("non-maximum suppression value"),DefaultValue(to_string(NMS_THRESH)));
    parser::ADD_ARG_INT("batchsize",Desc("batch size for input"),DefaultValue("1"));
    parser::ADD_ARG_STRING("enginefile",Desc("load from engine"),DefaultValue(""));

    //input
    parser::ADD_ARG_STRING("input",Desc("input image file"),DefaultValue(INPUT_IMAGE),ValueDesc("file"));
    parser::ADD_ARG_STRING("evallist",Desc("eval gt list"),DefaultValue(EVAL_LIST),ValueDesc("file"));

    if(argc < 2){
        parser::printDesc();
        exit(-1);
    }

    parser::parseArgs(argc,argv);


    
    batchSize = parser::getIntValue("batchsize"); 
    engineName =  parser::getStringValue("enginefile");
    if(engineName.length() > 0)
    {
        net.reset(new trtNet(engineName));
        assert(net->getBatchSize() == batchSize);
    }else{
		return 1;
	}
    

    outputCount = net->getOutputSize()/sizeof(float);
    // outputData = new float[outputCount];
	// outputData = std::make_unique<float[]>(outputCount);
	float * outputData_ = new float[outputCount];
	outputData.reset(outputData_);

    classNum = parser::getIntValue("class");
    c = parser::getIntValue("C");
    h = parser::getIntValue("H");
    w = parser::getIntValue("W");

    ros::spin();

    return 0;
}
