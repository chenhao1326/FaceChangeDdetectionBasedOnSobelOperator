#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>



using namespace std;  
using namespace cv;

/*���ģ����
 ����ֵΪģ���ȣ�ֵԽ��Խģ����ԽСԽ��������Χ��0����ʮ��10������Խ�������һ��Ϊ5��
 ����ʱ�����ⲿ�趨һ����ֵ��������ֵ����ʵ���������������ֵ������ֵ������ģ��ͼƬ��
 �㷨����ʱ����1������
*/
float VideoBlurDetect(const cv::Mat& srcimg)
{
	cv::Mat img;
	cv::cvtColor(srcimg, img, CV_BGR2GRAY); // �������ͼƬתΪ�Ҷ�ͼ��ʹ�ûҶ�ͼ���ģ����

	//ͼƬÿ���ֽ�������  
	int width = img.cols;
	int height = img.rows;
	ushort* sobelTable = new ushort[width * height];
	memset(sobelTable, 0, width * height * sizeof(ushort));

	int i, j, mul;
	//ָ��ͼ���׵�ַ  
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
	//����ͼƬ  
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
	//ģ����
	float result = (float)totLen / totCount;
	delete[] sobelTable;
	sobelTable = NULL;

	return result;
}

int main()
{
	Mat image, image_gray;       //��������Mat���������ڴ洢ÿһ֡��ͼ��

	image = imread("E:\\face\\Project1\\33.png");
	imshow("ԭͼ", image);

	//waitKey(0);

	cvtColor(image, image_gray, CV_BGR2GRAY);   //תΪ�Ҷ�ͼ
	equalizeHist(image_gray, image_gray);       //ֱ��ͼ��������ǿ�Աȶȷ��㴦��

	CascadeClassifier eye_Classifier;            //���������
	CascadeClassifier face_cascade;              //���������


	//���ط���ѵ������OpenCV�ٷ��ĵ���xml�ĵ�������ֱ�ӵ���
	//�ҵ�xml��·��D:\OpenCV\opencv\build\etc\haarcascades  

	if (!eye_Classifier.load("E:\\face\\Project1\\haarcascade_eye.xml"))    //��xml�ĵ����Ƶ��˵�ǰ��Ŀ��·����
	{
		cout << "����haarcascade_eye.xmlʱ���� !" << endl;
		return 0;

	}

	if (!face_cascade.load("E:\\face\\Project1\\haarcascade_frontalface_alt.xml"))    //��xml�ĵ����Ƶ��˵�ǰ��Ŀ��·����
	{
		cout << "����haarcascade_frontalface_alt.xmlʱ���� !" << endl;
		return 0;

	}


	//vector �Ǹ���ģ�� ��Ҫ�ṩ��ȷ��ģ��ʵ��  vector<Rect>���Ǹ�ȷ������ ģ���ʵ����

	vector<Rect> eyeRect;
	vector<Rect> faceRect;


	//����۾���λ��
	eye_Classifier.detectMultiScale(image_gray, eyeRect, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for (size_t eyeIdx = 0; eyeIdx < eyeRect.size(); eyeIdx++)
	{

		rectangle(image, eyeRect[eyeIdx], Scalar(0, 0, 255));    //�þ��λ�����⵽���۾���λ��

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
	���������壺
	const Mat& image: ��Ҫ������ͼ�񣨻Ҷ�ͼ��
	vector<Rect>& objects: ���汻����������λ����������
	double scaleFactor: ÿ��ͼƬ���ŵı���
	int minNeighbors: ÿһ����������Ҫ��⵽���ٴβ������������
	int flags�� ���������ŷ���������⣬��������ͼ��
	Size(): ��ʾ�����������С�ߴ�

	*/

	//������������λ��

	face_cascade.detectMultiScale(image_gray, faceRect, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for (size_t i = 0; i < faceRect.size(); i++)
	{
		rectangle(image, faceRect[i], Scalar(0, 0, 255));           //�þ��λ�����⵽������λ��
	}
	/*Rect rect(faceRect[0].tl().x, faceRect[0].tl().y, faceRect[0].width, faceRect[0].height);
	Mat ROI = image(rect);
	imshow("ROI_WIN", ROI);*/
	imshow("����ʶ��", image);         //��ʾ��ǰ
	waitKey(0);
	float result1 = VideoBlurDetect(image);
	float result2 = VideoBlurDetect(ROI);
	float result3 = VideoBlurDetect(ROI1);

	if (result2 <= result1+1 && result3 <= result1+1) {
		cout << "���Ǽ��� !" << endl;
	}
	else {
		cout << "�Ǽ��� !" << endl;
	}
	printf("%f\n",result1);
	printf("%f\n", result2);
	printf("%f\n", result3);

	return 0;

}
