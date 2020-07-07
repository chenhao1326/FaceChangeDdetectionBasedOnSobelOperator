#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>



using namespace std;  
using namespace cv;

/*检测模糊度
 返回值为模糊度，值越大越模糊，越小越清晰，范围在0到几十，10以下相对较清晰，一般为5。
 调用时可在外部设定一个阀值，具体阈值根据实际情况决定，返回值超过阀值当作是模糊图片。
 算法所耗时间在1毫秒内
*/
float VideoBlurDetect(const cv::Mat& srcimg)
{
	cv::Mat img;
	cv::cvtColor(srcimg, img, CV_BGR2GRAY); // 将输入的图片转为灰度图，使用灰度图检测模糊度

	//图片每行字节数及高  
	int width = img.cols;
	int height = img.rows;
	ushort* sobelTable = new ushort[width * height];
	memset(sobelTable, 0, width * height * sizeof(ushort));

	int i, j, mul;
	//指向图像首地址  
	uchar* udata = img.data;
	for (i = 1, mul = i * width; i < height - 1; i++, mul += width)
		for (j = 1; j < width - 1; j++)

			sobelTable[mul + j] = abs(udata[mul + j - width - 1] + 2 * udata[mul + j - 1] + udata[mul + j - 1 + width] - \
				udata[mul + j + 1 - width] - 2 * udata[mul + j + 1] - udata[mul + j + width + 1]);

	for (i = 1, mul = i * width; i < height - 1; i++, mul += width)
		for (j = 1; j < width - 1; j++)
			if (sobelTable[mul + j] < 50 || sobelTable[mul + j] <= sobelTable[mul + j - 1] || \
				sobelTable[mul + j] <= sobelTable[mul + j + 1]) sobelTable[mul + j] = 0;

	int totLen = 0;
	int totCount = 1;

	uchar suddenThre = 50;
	uchar sameThre = 3;
	//遍历图片  
	for (i = 1, mul = i * width; i < height - 1; i++, mul += width)
	{
		for (j = 1; j < width - 1; j++)
		{
			if (sobelTable[mul + j])
			{
				int   count = 0;
				uchar tmpThre = 5;
				uchar max = udata[mul + j] > udata[mul + j - 1] ? 0 : 1;

				for (int t = j; t > 0; t--)
				{
					count++;
					if (abs(udata[mul + t] - udata[mul + t - 1]) > suddenThre)
						break;

					if (max && udata[mul + t] > udata[mul + t - 1])
						break;

					if (!max && udata[mul + t] < udata[mul + t - 1])
						break;

					int tmp = 0;
					for (int s = t; s > 0; s--)
					{
						if (abs(udata[mul + t] - udata[mul + s]) < sameThre)
						{
							tmp++;
							if (tmp > tmpThre) break;
						}
						else break;
					}

					if (tmp > tmpThre) break;
				}

				max = udata[mul + j] > udata[mul + j + 1] ? 0 : 1;

				for (int t = j; t < width; t++)
				{
					count++;
					if (abs(udata[mul + t] - udata[mul + t + 1]) > suddenThre)
						break;

					if (max && udata[mul + t] > udata[mul + t + 1])
						break;

					if (!max && udata[mul + t] < udata[mul + t + 1])
						break;

					int tmp = 0;
					for (int s = t; s < width; s++)
					{
						if (abs(udata[mul + t] - udata[mul + s]) < sameThre)
						{
							tmp++;
							if (tmp > tmpThre) break;
						}
						else break;
					}

					if (tmp > tmpThre) break;
				}
				count--;

				totCount++;
				totLen += count;
			}
		}
	}
	//模糊度
	float result = (float)totLen / totCount;
	delete[] sobelTable;
	sobelTable = NULL;

	return result;
}

int main()
{
	Mat image, image_gray;       //定义两个Mat变量，用于存储每一帧的图像

	image = imread("E:\\face\\Project1\\33.png");
	imshow("原图", image);

	//waitKey(0);

	cvtColor(image, image_gray, CV_BGR2GRAY);   //转为灰度图
	equalizeHist(image_gray, image_gray);       //直发图均化，增强对比度方便处理

	CascadeClassifier eye_Classifier;            //载入分类器
	CascadeClassifier face_cascade;              //载入分类器


	//加载分类训练器，OpenCV官方文档的xml文档，可以直接调用
	//我的xml的路径D:\OpenCV\opencv\build\etc\haarcascades  

	if (!eye_Classifier.load("E:\\face\\Project1\\haarcascade_eye.xml"))    //把xml文档复制到了当前项目的路径下
	{
		cout << "导入haarcascade_eye.xml时出错 !" << endl;
		return 0;

	}

	if (!face_cascade.load("E:\\face\\Project1\\haarcascade_frontalface_alt.xml"))    //把xml文档复制到了当前项目的路径下
	{
		cout << "导入haarcascade_frontalface_alt.xml时出错 !" << endl;
		return 0;

	}


	//vector 是个类模板 需要提供明确的模板实参  vector<Rect>则是个确定的类 模板的实例化

	vector<Rect> eyeRect;
	vector<Rect> faceRect;


	//检测眼睛的位置
	eye_Classifier.detectMultiScale(image_gray, eyeRect, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for (size_t eyeIdx = 0; eyeIdx < eyeRect.size(); eyeIdx++)
	{

		rectangle(image, eyeRect[eyeIdx], Scalar(0, 0, 255));    //用矩形画出检测到的眼睛的位置

	}
	Rect rect(eyeRect[1].tl().x, eyeRect[1].tl().y, eyeRect[1].width, eyeRect[1].height);
	Mat ROI = image(rect);
	imshow("ROI_WIN", ROI);
	Rect rect1(eyeRect[2].tl().x, eyeRect[2].tl().y, eyeRect[2].width, eyeRect[2].height);
	Mat ROI1 = image(rect1);
	imshow("ROI_WIN1", ROI1);
	/*
	CV_WRAP virtual void detectMultiScale(
								   const Mat& image,
								   CV_OUT vector<Rect>& objects,
								   double scaleFactor=1.1,
								   int minNeighbors=3, int flags=0,
								   Size minSize=Size(),
								   Size maxSize=Size()
								   );
	各参数含义：
	const Mat& image: 需要被检测的图像（灰度图）
	vector<Rect>& objects: 保存被检测出的人脸位置坐标序列
	double scaleFactor: 每次图片缩放的比例
	int minNeighbors: 每一个人脸至少要检测到多少次才算是真的人脸
	int flags： 决定是缩放分类器来检测，还是缩放图像
	Size(): 表示人脸的最大最小尺寸

	*/

	//检测关于脸部的位置

	face_cascade.detectMultiScale(image_gray, faceRect, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for (size_t i = 0; i < faceRect.size(); i++)
	{
		rectangle(image, faceRect[i], Scalar(0, 0, 255));           //用矩形画出检测到脸部的位置
	}
	/*Rect rect(faceRect[0].tl().x, faceRect[0].tl().y, faceRect[0].width, faceRect[0].height);
	Mat ROI = image(rect);
	imshow("ROI_WIN", ROI);*/
	imshow("人脸识别", image);         //显示当前
	waitKey(0);
	float result1 = VideoBlurDetect(image);
	float result2 = VideoBlurDetect(ROI);
	float result3 = VideoBlurDetect(ROI1);

	if (result2 <= result1+1 && result3 <= result1+1) {
		cout << "不是假脸 !" << endl;
	}
	else {
		cout << "是假脸 !" << endl;
	}
	printf("%f\n",result1);
	printf("%f\n", result2);
	printf("%f\n", result3);

	return 0;

}
