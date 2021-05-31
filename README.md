# Image Processing Course - Project

## License Plate Detector in C++ using OpenCV

![GitHub repo size](https://img.shields.io/github/repo-size/scottydocs/README-template.md)

   Technical University of Cluj-Napoca\
   Computer Science - 3rd Year\
  2021, Spring

## **Usage**

### Clone the repository

**``
git clone https://github.com/blasio99/LicensePlateDetector_CPP.git
``**

## **Description**

### Requirements

- Color images of vehicles (at different distances) with visible license plate are provided (the license plate can be a little bit rotated to the horizontal direction)
- There is implemented an algorithm to identify and mark the areas where the license plates are found

### Basic steps on the approach

1. Image acquiring
2. Grayscale conversion
3. Noise reduction
4. Image thresholding
5. Canny Edge detection algorithm
6. Connected Component Labeling
7. Choosing th best candidate

### STEP 1 & 2

Loading the image should be the easiest part, because of the functionalities offered by OpenCV.  
The image color is converted to grayscale while reading.  
Big advantage of grayscale images is that it saves up a lot of memory and it is easier to work with them.

### STEP 3 & 4

Reduce the noise present in the image using some gaussian filters (mathematical properties)

- Reduce unwanted information which deteriorates image quality.
- Apply Gaussian convolution (ID = G * IS) - implementation may vary

Apply thresholding process to obtain a binary image ( black and white image suitable for Canny algorithm)

### STEP 5

The steps of the Canny edge detection method are given below

1. Noise filtering through a Gaussian Kernel
2. Computing the gradient’s module and direction
3. Non-maxima suppression of the gradient’s module
4. Edge linking through adaptive hysteresis thresholding

```cpp
/*
*	The edge detection method proposed by Canny is based on the image gradient computation but in addition it tries to:
*		- maximize the signal-to-noise ratio for proper detection
*		- find a good localization of the edge points
*		- minimize the number of positive responses around a single edge (suppression of the gradient module non-maxims)
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
```

### STEP 6 & 7

Label the same neighbouring pixels that belong to the same object.

- for this process a binary image is needed ( output of the Canny algorithm )

Finally, from those components, select the suitable one based on some mathematical and logical principles and rules.

## Visualization

Algorithm presentation:

<img src="Project/Images/car_MH_contours.png"/>
<img src="Project/Images/car_MH_canny.png"/>

## **Contributing**

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.  

- Fork the Project  
- Create your Feature Branch ( **`git checkout -b feature/AmazingFeature`** )
- Commit your Changes ( **`git commit -m 'Add some AmazingFeature'`** )
- Push to the Branch ( **`git push origin feature/AmazingFeature`** )
- Open a Pull Request  

## **Contact**

- Benedek Balázs - [LinkedIn Profile](https://www.linkedin.com/in/balazs-benedek-009322183/)
- E-mail: benedekbalazs1999@gmail.com
- Project Link: [GitHub - License Plate Detector](https://github.com/blasio99/LicensePlateDetector_CPP)
