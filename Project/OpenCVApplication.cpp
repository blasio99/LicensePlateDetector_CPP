/***********************************************
*         Benedek Balazs - Group 30434
*                 25/04/2021
************************************************
*
*      Technical University of Cluj - Napoca
*    Faculty of Automation and Computer Science
*           Image Processing 2020/2021
*
*					Project
*
*     _     _            _        ____   ____
*    | |__ | | ___  ___ |_| ___  /    \ /    \
*    |    \| |/__ \| __|| |/   \ \__'  |\__'  |
*    |  .  | |  .  |__ || |  .  | __|  | __|  |
*    \____/|_|\___/|___||_|\___/ |____/ |____/
*
************************************************
*              github.com/blasio99
*
*/

#include "stdafx.h"
#include "common.h"
#include <stdlib.h>
#include <map>
#include <random>
#include <cmath>
#include <fstream>

#define N4 4
#define N8 8
#define NP 2
#define WAIT   true
#define NOWAIT false
#define DILATION 4
#define EROSION  5

#define WEAK_EDGE   127
#define STRONG_EDGE 255
#define NON_EDGE    0 

#define min3(a, b, c) min(a, min(b, c))
#define max3(a, b, c) max(a, max(b, c))

#define min4(a, b, c, d) min(a, min3(b, c, d))
#define max4(a, b, c, d) max(a, max3(b, c, d))

using namespace std;

typedef Mat_ <uchar> Image;
typedef Mat_ <Vec3b> ColorImage;
typedef Mat_ <int> StructuringElement;

// -------------------------------------------------- Lab 1 ---------------------------------------------------------

void imageShow(char* name, Mat img, boolean toWait) {
	imshow(name, img);
	if (toWait) waitKey(0);
}

bool isInside(Mat img, int i, int j) {
	if (i < 0 || i >= img.rows) return false;
	if (j < 0 || j >= img.cols) return false;
	return true;
}

/*
*	Creates the the structuring elements, the Mat_ <int> of the neighbourhood
*
*	@param type : the type of neighbourhood (4-type or 8-type)
*	@return the cross structure (non-diagonal neighbours) or square structure (8 neigbours, diagonal, horizontal and vertical)
*/
StructuringElement getStructuringElement(int type) {
	StructuringElement CROSS = (StructuringElement(3, 3) << 255, 0, 255, 0, 0, 0, 255, 0, 255);
	StructuringElement SQUARE = (StructuringElement(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0);

	switch (type) {
	case N8: // 8-type neighbourhood
		return SQUARE;
	case N4: // 4-type neighbourhood
		return CROSS;
	}
}

/*
*	It changes the current pixel to background color or not, together with its neighbours (in this case 8-type neighbours)
*
*	@param img  : the input image
*	@param dst  : the output image with modified pixels
*	@param i	: the x coordinate of the current pixel
*	@param j	: the y coordiante of the current pixel
*	@param type	: the type of morphological operation (DILATION or EROSION)
*	@return		: the new image with modified pixels
*/
Image pixelModification(Image img, Image dst, int i, int j, int type) {
	int di[5] = { -1, 1, 0, 0, 0 };
	int dj[5] = { 0, 0, -1, 1, 0 };

	StructuringElement structElement = getStructuringElement(N8);

	// Dilation or Erosion
	switch (type) {
	case DILATION:
		for (int x = 0; x < structElement.rows; ++x)
			for (int y = 0; y < structElement.cols; ++y)
				// Neighbourhood modification
				if (isInside(img, i + x - structElement.rows / 2, j + y - structElement.cols / 2) && structElement(x, y) == 0)
					dst(x + i - structElement.rows / 2, y + j - structElement.cols / 2) = 0;

		break;
	case EROSION:
		for (int x = 0; x < structElement.rows; ++x)
			for (int y = 0; y < structElement.cols; ++y)
				// Neighbourhood modification
				if (isInside(img, i + x - structElement.rows / 2, j + y - structElement.cols / 2) && structElement(x, y) == 0)
					if (img(x + i - structElement.rows / 2, y + j - structElement.cols / 2) == 255)
						dst(i, j) = 255;
		break;
	}
	return dst;
}

/*
*	The dilation process is performed by laying the structuring element B on the image A and sliding it across the image
*	from left to right, top to bottom.
*	The result image has the same size as image A. Its pixels are initialized to ‘background’.
*
*	@param img : the grayscale image on which is applied the eroding
*	@return the new image with erosion on it
*/
Image dilation(Image img) {
	Image dst = img.clone();

	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
			if (img(i, j) == 0)
				dst = pixelModification(img, dst, i, j, DILATION);

	return dst;
}

/*
*	The erosion process is similar to dilation, but the effect is somehow opposite
*	The result image is initialized to 'background'
*
*	@param img : the grayscale image on which is applied the eroding
*	@return the new image with erosion on it
*/
Image erosion(Image img) {
	Image dst = img.clone();

	for (int i = 1; i < img.rows - 1; ++i)
		for (int j = 1; j < img.cols - 1; ++j)
			dst = pixelModification(img, dst, i, j, EROSION);

	return dst;
}

/*
*	Closing consists of a dilation followed by erosion and can be used to fill in holes and small gaps
*
*	@param img : the grayscale image on which the closing is applied
*	@return the new image with closing applied on it
*/
Image closing(Image img) {
	return dilation(erosion(img));;
}

/*
*	Gaussian convolution kernel construction
*
*	@param sigma : The Gaussian noise rate
*	@return a matrix of floats with the Gaussian values
*/
Mat_ <float> construct_gaussian_2Dkernel(float sigma) {

	int w = sigma * 6 + 1;

	Mat_ <float> gauss(w, w, CV_32FC1);

	printf("\nKernel: \n\n");

	/*
						(x - x0)² + (y - y0)²
				1     - ---------------------
	G(x, y) = ----- e            2σ²
			   2πσ²
	*/
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < w; j++) {
			float exponent = -((i - w / 2) * (i - w / 2) + (j - w / 2) * (j - w / 2)) / (2 * sigma * sigma);
			float val = 1 / (2.0 * PI * sigma * sigma) * exp(exponent);
			printf("%f ", val);
			gauss.at<float>(i, j) = val;
		}
		printf("\n");
	}
	return gauss;
}

/*
*	Gaussian 1D filtering by the convolution built previously
*
*	@param img   : the input image being filtered
*	@param sigma : the Gaussian noise rate
*	@return the filtered image
*/
Image gaussian_1D_filtering(Image img, float sigma) {
	int w = sigma * 6 + 1;
	Mat_<float> kernel = construct_gaussian_2Dkernel(sigma);

	vector<float> row, col;

	for (int i = 0; i < w; i++) {
		row.push_back(kernel.at<float>((int)(w / 2), i));
		col.push_back(kernel.at<float>(i, (int)(w / 2)));
	}

	// the rows of the kernel
	printf("\nRow: ");
	float rows = 0;
	for (int i = 0; i < w; i++) {
		printf("%f, ", row.at(i));
		rows += row.at(i);
	}
	// the columns of the kernel
	printf("\nCol: ");
	float cols = 0;
	for (int i = 0; i < w; i++) {
		printf("%f, ", col.at(i));
		cols += col.at(i);
	}

	// destination, resulting image
	Image dst = img.clone();
	// an auxiliar image, helper
	Image aux = img.clone();

	double t = (double)getTickCount(); // Get the current time [ms]

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			float sum = 0;
			for (int u = 0; u < w; u++)
				if (isInside(img, i, j + u - (w / 2)))
					sum += img(i, j + u - (w / 2)) * row.at(u);

			aux(i, j) = sum / rows;
		}
	}
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			float sum = 0;
			for (int v = 0; v < w; v++)
				if (isInside(img, i + v - (w / 2), j))
					sum += aux(i + v - (w / 2), j) * col.at(v);


			dst(i, j) = sum / cols;
		}
	}

	// time counter for operations
	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("\nTime = %.3f [ms]\n", t * 1000);

	return dst;
}

/*
*	Using Sobel convolution arrays as input (in this case) the gradient is being approximated
*
*	@param src   : input image to filter
*	@param array : the gradient array (gradient x or gradient y)
*	@return the newly created destination image, which is a copy of the source image with modificated pixels
*/
Mat_ <float> gen_filter(Image src, Mat_ <float> array) {
	Mat_ <float> dst(src.rows, src.cols);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			float sum = 0;
			for (int u = 0; u < array.rows; u++) {
				for (int v = 0; v < array.cols; v++) {
					if (isInside(src, i + u - (array.rows - 2), j + v - (array.rows / 2)))
						sum += src(i + u - (array.rows - 2), j + v - (array.rows / 2)) * array(u, v);
				}
			}
			dst(i, j) = sum;
		}
	}
	return dst;
}

/*
*	It computes and gives the region, the quarter of the circle where the orientation is
*
*	@param phi : the orientation
*	@return the region of the s=circle (1,2,3 or 4)
*/
int getRegion(float phi) {
	float norm = CV_PI / 8.0;
	if ((phi <= norm && phi >= -norm) || (phi >= 7 * norm || phi <= -7 * norm))
		return 0;
	else if ((phi <= 3 * norm && phi >= norm) || (phi <= -5 * norm && phi >= -7 * norm))
		return 1;
	else if ((phi <= 5 * norm && phi >= 3 * norm) || (phi <= -3 * norm && phi >= -5 * norm))
		return 2;
	else if ((phi <= 7 * norm && phi >= 5 * norm) || (phi <= -norm && phi >= -3 * norm))
		return 3;
}

/*
*	The edge detection method proposed by Canny is based on the image gradient computation but in addition it tries to:
*		- maximize the signal-to-noise ratio for proper detection
*		- find a good localization of the edge points
*		- minimize the number of positive responses around a single edge (suppression of the gradient module non-maxims)
*
*	The steps of the Canny edge detection method are the following:
*		1. Noise filtering through 1D Gaussian kernel
*		2. Computing the gradient's module and direction
*		3. Non-maxima supression of the gradient's module
*		4. Edge linking through adaptive hysterisis thresholding
*
*	@param src : the source grayscale image on which is applied the Canny edge detection method
*	@return the final image with detected edges
*/
Image canny_edge_detection(Image src) {

	// Noise reduction through Gaussian kernel
	Image gaussianFilter = gaussian_1D_filtering(src, 0.7);

	// Computing the gradient's module
	// Sobel convolutions - 11.4
	float gradXArray[] = { -1, 0, 1, -2,  0,  2, -1,  0,  1 };
	float gradYArray[] = { 1, 2, 1,  0,  0,  0, -1, -2, -1 };

	Mat_ <float> gradX(3, 3, gradXArray);
	Mat_ <float> gradY(3, 3, gradYArray);

	Mat_ <float> gX = gen_filter(gaussianFilter, gradX);
	Mat_ <float> gY = gen_filter(gaussianFilter, gradY);

	Mat_ <float> grd(src.rows, src.cols);
	Image magn = src.clone();
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			grd(i, j) = sqrt(gX(i, j) * gX(i, j) + gY(i, j) * gY(i, j)) / (4 * sqrt(2));
			// to uchar
			magn(i, j) = (uchar)grd(i, j);
		}

	// [DEBUG]
	// imageShow("grad", magn, NOWAIT);

	// Computing the gradient's direction (orientation)
	// calculate orientation (11.7)
	Mat_ <float> orientation(src.rows, src.cols, CV_32FC1);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			orientation.at<float>(i, j) = atan2(gY.at<float>(i, j), gX.at<float>(i, j));
		}
	}

	Mat_ <float> nms = grd.clone();
	Mat_ <float> gs = grd.clone();
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			float phi = orientation.at<float>(i, j);
			int region = getRegion(phi);

			switch (region) {
			case 0: {
				if ((isInside(nms, i, j - 1) && nms.at<float>(i, j) < nms.at<float>(i, j - 1)) ||
					(isInside(nms, i, j + 1) && nms.at<float>(i, j) < nms.at<float>(i, j + 1)))
					gs.at<float>(i, j) = 0;
			} break;
			case 1: {
				if ((isInside(nms, i - 1, j + 1) && nms.at<float>(i, j) < nms.at<float>(i - 1, j + 1)) ||
					(isInside(nms, i + 1, j - 1) && nms.at<float>(i, j) < nms.at<float>(i + 1, j - 1)))
					gs.at<float>(i, j) = 0;
			} break;
			case 2: {
				if ((isInside(nms, i - 1, j) && nms.at<float>(i, j) < nms.at<float>(i - 1, j)) ||
					(isInside(nms, i + 1, j) && nms.at<float>(i, j) < nms.at<float>(i + 1, j)))
					gs.at<float>(i, j) = 0;
			} break;
			case 3: {
				if ((isInside(nms, i - 1, j - 1) && nms.at<float>(i, j) < nms.at<float>(i - 1, j - 1)) ||
					(isInside(nms, i + 1, j + 1) && nms.at<float>(i, j) < nms.at<float>(i + 1, j + 1)))
					gs.at<float>(i, j) = 0;
			} break;
			}

		}
	}
	Image maxim = src.clone();
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			maxim(i, j) = (uchar)gs.at<float>(i, j);
		}
	}

	// [DEBUG]
	// imageShow("nms", maxim, NOWAIT);

	// Lab 12 - canny edge detection algorithm from maxima supression
	// Edge linking through adaptive hysteresis thresholding.
	// Image thresholding
	int hist[256] = { 0 };
	float p = 0.25;
	for (int i = 0; i < maxim.rows; i++)
		for (int j = 0; j < maxim.cols; j++)
			hist[maxim(i, j)]++;

	int edgePixels = p * ((src.rows - 2) * (src.cols - 2) - hist[0]);

	printf("edgePixels: %d\n", edgePixels);
	int sum = 0;
	int threshold = 0;

	for (int i = 255; i > 0; i--) {
		sum += hist[i];
		if (sum > edgePixels) {
			threshold = i;
			break;
		}
	}

	int thresholdHigh = threshold;
	float k = 0.4; // lab example
	int thresholdLow = k * threshold;

	// Weak edges are eliminated
	// Labels higher than thresholdHigh are labeled as STRONG_EDGES
	// Labels between thersholdLow and thresholdHigh are labeled WEAK_EDGES
	// Labels below thresholdLow are considered NON_EDGES and are rejected
	Image adapt = maxim.clone();
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			adapt(i, j) = maxim(i, j) < thresholdLow ? NON_EDGE /*0*/ : (maxim(i, j) >= thresholdHigh ? STRONG_EDGE /*255*/ : WEAK_EDGE /*127*/);

	// [DEBUG]
	// imageShow("adaptive", adapt, NOWAIT);

	Image finalMat = adapt.clone();
	// 8-type neighbourhood is used
	int di[8] = { -1, -1, -1,  0,  0,  1,  1,  1 };
	int dj[8] = { -1,  0,  1, -1,  1, -1,  0,  1 };

	// Scanning the the image, top left to bottom right 
	for (int i = 0; i < finalMat.rows; i++) {
		for (int j = 0; j < finalMat.cols; j++) {
			// Pickig the first STRONG_EDGE point encountered
			if (finalMat(i, j) == STRONG_EDGE)
			{
				queue<Point2i> Q;
				// Pushing its coordinates to queue
				Q.push(Point2i(j, i));

				// While the queue is not empty
				while (!Q.empty())
				{
					// Extracting the first point from the queue
					Point q = Q.front();
					Q.pop();

					// Labeling in the image all the neighbours of the current point
					for (int k = 0; k < 8; k++)
						if (isInside(finalMat, q.y + di[k], q.x + dj[k]) && finalMat(q.y + di[k], q.x + dj[k]) == WEAK_EDGE) {
							Q.push(Point2i(q.x + dj[k], q.y + di[k]));
							finalMat(q.y + di[k], q.x + dj[k]) = STRONG_EDGE;
						}
				}
			}
		}
	}
	// Eliminating the ramining WEAK_EDGE points from the image by turning them to NON_EDGE (0)
	for (int i = 0; i < finalMat.rows; i++)
		for (int j = 0; j < finalMat.cols; j++)
			if (finalMat(i, j) == WEAK_EDGE)
				finalMat(i, j) = 0;


	imageShow("Canny edge detection", finalMat, NOWAIT);

	return finalMat;
}


/**
*	Get the minimal area -> minHeight * minWidth, for which every candidate will be resized
*
*	@param candidates : a vector of possible license plate candidate grayscale images
*	@return a structure of type Size with minimum width and minimum height
*/
Size minArea(vector<Image> candidates) {
	int minHeight = candidates.at(0).rows;
	int minWidth = candidates.at(0).cols;

	for (int i = 1; i < candidates.size(); i++) {
		Image c = candidates.at(i);

		if (c.cols * c.rows < minHeight * minWidth) {
			minHeight = c.rows;
			minWidth = c.cols;
		}
	}
	return Size(minHeight, minWidth);
}

/**
*	Comparator for the std::min_element function where we compare them by the sum of their values
*/
struct VerticalProjectionComparator
{
	bool operator ()(const Mat& m1, const Mat& m2)
	{
		return (sum(m1)[0] / m1.cols * 1.0) > (sum(m2)[0] / m2.cols * 1.0);
	}
};

/**
*	Creating the candidate list and choosing the best fit
*
*	@param candidates : a grayscale image (Mat_ <uchar>) vector with license plate candidates
*	@return the candidate plate (grayscale image)
*/
Image verticalProjection(vector<Image> candidates) {
	Size newSize = minArea(candidates);
	int index = -1;

	vector<Image> cs;
	for (int i = 0; i < candidates.size(); i++) {
		/* [DEBUG]
		// displaying the possible license plates
		char* name = (char*)malloc(6);
		sprintf(name, "name%d", i);
		printf("%d -> %d, %d\n", i, candidates.at(i).rows, candidates.at(i).cols);
		imageShow(name, candidates.at(i), NOWAIT);
		*/
		Image candidate;
		resize(candidates.at(i), candidate, newSize);
		cs.push_back(candidate);
		int iMinus1 = i - 1;
		if (i > 0 && abs(candidates.at(i).rows - candidates.at(iMinus1).rows) <= 1 &&
			abs(candidates.at(i).cols - candidates.at(iMinus1).cols) <= 1)
			return candidates.at(i);
	}
	// creates an image with the smallest candidate from 'cs' 
	Image licensePlate = *min_element(cs.begin(), cs.end(), VerticalProjectionComparator());

	for (int i = 0; i < cs.size(); i++)
		if (cs[i].rows == licensePlate.rows && cs[i].rows == licensePlate.rows && sum(cs[i]) == sum(licensePlate))
			index = i;

	return candidates.at(index);
}
/*
*	Extracting the candidate plate from the image by getting the contour parameters
*
*	@param filtered : the filtered image from where we extract the plate
*	@param possible : the possible license plate coordinates from the source image
*	@param i		: the index in the vector where the candidate plates' coordinates are
*	@return a grayscale image of the candidate plate
*/
Image get_candidate_plate(Image filtered, vector<vector<Point>> possible, int i) {
	vector<Point> plate;
	approxPolyDP(possible[i], plate, 0.015 * arcLength(possible[i], true), true);

	//extracting the plate like a rectangle, finding the xMin, yMin, xMax, yMax
	/*

			|
	   yMax	|   ----------------
			|   |  |  |  |''\  |
			|   |  |''|  |''\  |
	   yMin	|   ----------------
			|_______________________
			   xMin			  xMax
	*/
	int xMin = min4(plate[0].x, plate[1].x, plate[2].x, plate[3].x);
	int xMax = max4(plate[0].x, plate[1].x, plate[2].x, plate[3].x);
	int yMin = min4(plate[0].y, plate[1].y, plate[2].y, plate[3].y);
	int yMax = max4(plate[0].y, plate[1].y, plate[2].y, plate[3].y);

	int height = yMax - yMin;
	int width = xMax - xMin;

	//creating the license plate image
	Image plateCandidate(height, width);
	for (int i = yMin; i < yMax; i++) {
		for (int j = xMin; j < xMax; j++)
			plateCandidate(i - yMin, j - xMin) = filtered(i, j);
	}
	return plateCandidate;
}

/*
*	Project driver
*/
int main()
{
	// Choosing the image from folder
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		// Image read as grayscale => no need for manual conversion
		Image src = imread(fname, IMREAD_GRAYSCALE);

		// Correction of some unwanted pixels with other colors
		src = closing(src);

		// CANNY EDGE DETECTION
		Image edges(src.cols, src.rows);
		edges = canny_edge_detection(src);

		vector<vector<Point>> contours;
		vector<vector<Point>> possible;
		vector<Vec4i> hierarchy;
		// Find contours method implemented in OpenCV library
		findContours(edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

		// Approximating a curve or a polygon with another curve/polygon with less vertices 
		// so that the distance between them is less or equal to the specified precision
		// in this case we need RECTANGLES
		for (int i = 0; i < contours.size(); i++) {
			vector<Point> approx;

			approxPolyDP(contours[i], approx, 0.015 * arcLength(contours[i], true), true);

			if (approx.size() == 4) // rectangle
				possible.push_back(contours[i]);
		}

		// Recreating the color image (destination image)
		ColorImage dst(src.rows, src.cols);
		cvtColor(src, dst, COLOR_GRAY2RGB);

		vector<Image> candidates;
		for (int i = 0; i < possible.size(); i++) {
			// drawing the find contours
			drawContours(dst, possible, i, Scalar(0, 0, 255), 2, 8, hierarchy, 0, Point());

			// getting in every step the possible candidate plate
			Image candidatePlate = get_candidate_plate(src /*filtered*/, possible, i);

			int height = candidatePlate.rows;
			int width = candidatePlate.cols;
			int fullHeight = src.rows;
			int fullWidth = src.cols;

			double hr = fullHeight / height * 1.0;
			double wr = fullWidth / width * 1.0;

			// [EXPERIMENT]
			// modify these ratios to get bigger spectre / better results
			// full / plate axis ratios
			if (hr > 3.5 && wr > 1.5) {
				// for the license plate ratio
				if (height > 15 && width > 35)
					if (width > 1.5 * height && width < 6 * height)
						candidates.push_back(candidatePlate);

			}
		}
		// show the drawn contours on the source image
		imageShow("contours", dst, NOWAIT);

		// [DEBUG]
		// The list size of candidate plates 
		//printf("\n\ncandidates list size: %d\n", (int)candidates.size());

		if (candidates.size() > 0) {
			Image licensePlate = verticalProjection(candidates);
			imageShow("license plate", licensePlate, WAIT);
		}
		else {
			// if there is no plate anticipated
			printf("There is no license plate found");
			waitKey(0);
		}
	}
	return 0;
}
