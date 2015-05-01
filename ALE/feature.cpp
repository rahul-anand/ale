#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "feature.h"
#include "potential.h"

LFeature::LFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension)
{
	dataset = setDataset;
	evalFolder = setEvalFolder, evalExtension = setEvalExtension, trainFolder = setTrainFolder, trainFile = setTrainFile;
}

#ifdef MULTITHREAD

struct LEvaluateFeatureParams
{
	LFeature *feature;
	char *fileName;
};

thread_return EvaluateFeatureThread(void *par)
{
	LEvaluateFeatureParams *params = (LEvaluateFeatureParams *)par;
	params->feature->Evaluate(params->fileName);
	return(thread_defoutput);
}
#endif

void LFeature::Evaluate(LList<char *> &imageFiles, int from, int to)
{
	if(to == -1) to = imageFiles.GetCount();
#ifdef MULTITHREAD
	int i;
	int processors = GetProcessors(), running = 0, ind = from;
	thread_type *threads;
	LEvaluateFeatureParams *params;

	threads = new thread_type[processors];	
	memset(threads, 0, processors * sizeof(thread_type));
	params = new LEvaluateFeatureParams[processors];
	InitializeCriticalSection();

	for(i = 0; i < processors; i++) if(ind < to)
	{
		params[i].feature = this, params[i].fileName = imageFiles[ind];
		threads[i] = NewThread(EvaluateFeatureThread,  &params[i]);
		if(threads[i] != 0)
		{
			printf("Evaluating feature for image %d..\n", ind);
			running++;
			ind++;
		}
	}
	
	while(running)
	{
		for(i = 0; i < processors; i++) if(threads[i])
		{
		        if(ThreadFinished(threads[i]))
			{
				CloseThread(&threads[i]);
				running--;

				if(ind < to)
				{
					params[i].fileName = imageFiles[ind];
					threads[i] = NewThread(EvaluateFeatureThread,  &params[i]);
					if(threads[i] != 0)
					{
						printf("Evaluating feature for image %d..\n", ind);
						running++;
						ind++;
					}
				}
			}
		}
		Sleep(0);
	}
	DeleteCriticalSection();

	delete[] threads;
	delete[] params;
#else
	for(int i = from; i < to; i++)
	{
		printf("Evaluating feature for image %d..\n", i);
		Evaluate(imageFiles[i]);
	}
#endif
}

LDenseFeature::LDenseFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension) : LFeature(setDataset, setTrainFolder, setTrainFile, setEvalFolder, setEvalExtension)
{
}

void LDenseFeature::Evaluate(char *imageFileName)
{
#ifdef MULTITHREAD
	EnterCriticalSection();
#endif
	char *fileName;
	fileName = GetFileName(dataset->imageFolder, imageFileName, dataset->imageExtension);
	LLabImage labImage(fileName);
	delete[] fileName;
#ifdef MULTITHREAD
	LeaveCriticalSection();
#endif
	LImage<unsigned short> image;
	Discretize(labImage, image, imageFileName);

	fileName = GetFileName(evalFolder, imageFileName, evalExtension);
	image.Save(fileName);
	delete[] fileName;
}

LTextonFeature::LTextonFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, double setFilterScale, int setSubSample, int setNumberOfClusters, double setKMeansMaxChange, int setPointsPerKDTreeCluster) : LDenseFeature(setDataset, setTrainFolder, setTrainFile, setEvalFolder, setEvalExtension)
{
	subSample = setSubSample;
	filterScale = setFilterScale;
	numberOfClusters = setNumberOfClusters, kMeansMaxChange = setKMeansMaxChange, pointsPerKDTreeCluster = setPointsPerKDTreeCluster;
	CreateFilterList();
	clustering = new LKMeansClustering<double>(setTrainFolder, setTrainFile, GetTotalFilterCount(), numberOfClusters, kMeansMaxChange, pointsPerKDTreeCluster);
}

LTextonFeature::~LTextonFeature()
{
	for(int i = 0; i < filters.GetCount(); i++) delete filters[i];
	delete clustering;
}

void LTextonFeature::Train(LList<char *> &trainImageFiles)
{
	int fileSum = trainImageFiles.GetCount(), totalSize = 0, i, j;
	double *filterData, *dataTo;

	int filterCount = GetTotalFilterCount();
	printf("Counting total filter response size..\n");

	for(i = 0; i < trainImageFiles.GetCount(); i++)
	{
		char *fileName;
		fileName = GetFileName(dataset->imageFolder, trainImageFiles[i], dataset->imageExtension);
		LRgbImage rgbImage(fileName);
		delete[] fileName;

		int subWidth = (rgbImage.GetWidth() + subSample - 1) / subSample;
		int subHeight = (rgbImage.GetHeight() + subSample - 1) / subSample;
		totalSize += subWidth * subHeight;
	}
	filterData = new double[totalSize * filterCount];
	dataTo = filterData;

	printf("Calculating sampled filter responses..\n");

	for(i = 0; i < trainImageFiles.GetCount(); i++)
	{
		char *fileName;
		fileName = GetFileName(dataset->imageFolder, trainImageFiles[i], dataset->imageExtension);
		LLabImage labImage(fileName);
		delete[] fileName;

		int subWidth = (labImage.GetWidth() + subSample - 1) / subSample;
		int subHeight = (labImage.GetHeight() + subSample - 1) / subSample;

		LImage<double> filterResponse(subWidth, subHeight, filterCount);

		int offset = 0;
		for(j = 0; j < filters.GetCount(); offset += filters[j]->GetBands(), j++) filters[j]->Filter(labImage, 0, filterResponse, offset, subSample);

		memcpy(dataTo, filterResponse.GetData(), subWidth * subHeight * filterCount * sizeof(double));
		dataTo += subWidth * subHeight * filterCount;
	}

	printf("Calculating K-Means clusters..\n");

	clustering->Cluster(filterData, totalSize);
	delete[] filterData;

	SaveTraining();
}

void LTextonFeature::CreateFilterList()
{
	double sigma;

	for(int i = 0; i < filters.GetCount(); i++) delete filters[i];
	filters.Clear();

	for(sigma = (double)1.0; sigma <= (double)4.0; sigma *= (double)2.0) filters.Add(new LGaussianFilter2D<double>(sigma * filterScale, 3));
	for(sigma = (double)1.0; sigma <= (double)8.0; sigma *= (double)2.0) filters.Add(new LLaplacianFilter2D<double>(sigma * filterScale, 1));

	for (sigma = (double)2.0; sigma <= (double)4.0; sigma *= (double)2.0)
	{
		filters.Add(new LGaussianDerivativeXFilter2D<double>(sigma * filterScale, (double)3.0 * sigma * filterScale, 1));
		filters.Add(new LGaussianDerivativeYFilter2D<double>((double)3.0 * sigma * filterScale, sigma * filterScale, 1));
	}
}

int LTextonFeature::GetTotalFilterCount()
{
	int filterCount = 0;
	for(int i = 0; i < filters.GetCount(); i++) filterCount += filters[i]->GetBands();
	return(filterCount);
}

void LTextonFeature::Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName)
{
	int i, j;
	int width = labImage.GetWidth();
	int height = labImage.GetHeight();

	LImage<double> filterResponse(width, height, GetTotalFilterCount());

	int offset = 0;
	for(j = 0; j < filters.GetCount(); offset += filters[j]->GetBands(), j++) filters[j]->Filter(labImage, 0, filterResponse, offset, 1);

	image.SetResolution(width, height, 1);
	for(i = 0; i < height; i++) for(j = 0; j < width; j++) image(j, i, 0) = clustering->NearestNeighbour(filterResponse(j, i));
}

void LTextonFeature::LoadTraining()
{
	clustering->LoadTraining();
}

void LTextonFeature::SaveTraining()
{
	clustering->SaveTraining();
}

int LTextonFeature::GetBuckets()
{
	return(clustering->GetClusters());
}

LLocationFeature::LLocationFeature(LDataset *setDataset, const char *setEvalFolder, const char *setEvalExtension, int setLocationBuckets) : LDenseFeature(setDataset, NULL, NULL, setEvalFolder, setEvalExtension)
{
	double min[2], max[2];
	min[0] = min[1] = 0, max[0] = max[1] = 1;
	clustering = new LLatticeClustering<double>(2, min, max, setLocationBuckets);
}

LLocationFeature::~LLocationFeature()
{
	if(clustering != NULL) delete clustering;
}

void LLocationFeature::Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName)
{
	int width = labImage.GetWidth();
	int height = labImage.GetHeight();
	image.SetResolution(width, height, 1);
	unsigned short *dataBucket = image.GetData();

	double val[2];
	for(int k = 0; k < height; k++) for(int j = 0; j < width; j++, dataBucket++) 
	{
		val[0] = j / (double)width;
		val[1] = k / (double)height;
		*dataBucket = clustering->NearestNeighbour(val);
	}
}

int LLocationFeature::GetBuckets()
{
	return(clustering->GetClusters());
}

LSiftFeature::LSiftFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setWindowSize, int setWindowNumber, int set360, int setAngles, int setSubSample, int setNumberOfClusters, double setKMeansMaxChange, int setPointsPerKDTreeCluster, int setSymetric) : LDenseFeature(setDataset, setTrainFolder, setTrainFile, setEvalFolder, setEvalExtension)
{
	subSample = setSubSample, is360 = set360, angles = setAngles;
	numberOfClusters = setNumberOfClusters, kMeansMaxChange = setKMeansMaxChange, pointsPerKDTreeCluster = setPointsPerKDTreeCluster;
	sizeCount = 1;
	windowSize = new int[1];
	windowSize[0] = setWindowSize;
	windowNumber = setWindowNumber;
	clustering = new LKMeansClustering<double>(setTrainFolder, setTrainFile, angles * windowNumber * windowNumber, numberOfClusters, kMeansMaxChange, pointsPerKDTreeCluster);
	diff = !setSymetric;
}

LSiftFeature::LSiftFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setSizeCount, int *setWindowSize, int setWindowNumber, int set360, int setAngles, int setSubSample, int setNumberOfClusters, double setKMeansMaxChange, int setPointsPerKDTreeCluster, int setSymetric) : LDenseFeature(setDataset, setTrainFolder, setTrainFile, setEvalFolder, setEvalExtension)
{
	subSample = setSubSample, is360 = set360, angles = setAngles;
	numberOfClusters = setNumberOfClusters, kMeansMaxChange = setKMeansMaxChange, pointsPerKDTreeCluster = setPointsPerKDTreeCluster;
	sizeCount = setSizeCount;
	windowSize = new int[sizeCount];
	memcpy(windowSize, setWindowSize, sizeCount * sizeof(int));
	windowNumber = setWindowNumber;
	clustering = new LKMeansClustering<double>(setTrainFolder, setTrainFile, angles * windowNumber * windowNumber, numberOfClusters, kMeansMaxChange, pointsPerKDTreeCluster);
	diff = !setSymetric;
}

LSiftFeature::~LSiftFeature()
{
	delete clustering;
	delete[] windowSize;
}

void LSiftFeature::Train(LList<char *> &trainImageFiles)
{
	int fileSum = trainImageFiles.GetCount(), totalSize = 0, i, j, k, l, m, n, r;
	double *siftData, *dataTo;

	printf("Counting total sift response size..\n");

	for(i = 0; i < trainImageFiles.GetCount(); i++)
	{
		char *fileName;
		fileName = GetFileName(dataset->imageFolder, trainImageFiles[i], dataset->imageExtension);
		LRgbImage rgbImage(fileName);
		delete[] fileName;

		int subWidth = (rgbImage.GetWidth() + subSample - 1) / subSample;
		int subHeight = (rgbImage.GetHeight() + subSample - 1) / subSample;
		totalSize += subWidth * subHeight;
	}
	siftData = new double[totalSize * angles * windowNumber * windowNumber * sizeCount];
	dataTo = siftData;

	printf("Calculating sampled sift responses..\n");

	for(i = 0; i < trainImageFiles.GetCount(); i++)
	{
		char *fileName;
		fileName = GetFileName(dataset->imageFolder, trainImageFiles[i], dataset->imageExtension);
		LLabImage labImage(fileName);
		delete[] fileName;

		int width = labImage.GetWidth();
		int height = labImage.GetHeight();
		int subWidth = (width + subSample - 1) / subSample;
		int subHeight = (height + subSample - 1) / subSample;

		LImage<double> siftResponse(subWidth, subHeight, angles * windowNumber * windowNumber);

		LImage<double> diffImage(width, height, angles);
		memset(diffImage.GetData(), 0, width * height * angles * sizeof(double));

		double *diffData = diffImage.GetData();
		for(j = 0; j < height; j++) for(k = 0; k < width; k++, diffData += angles)
		{
			double dx, dy, mag, ang;
			dx = labImage((k == width - 1) ? k : k + 1, j, 0) - labImage((k == 0) ? 0 : k - diff, j, 0);
			dy = labImage(k, (j == height - 1) ? j : j + 1, 0) - labImage(k, (j == 0) ? 0 : j - diff, 0);

			mag = sqrt(dx * dx + dy * dy);
			ang = atan2(dy, dx);
			if(ang < 0) ang += 2 * LMath::pi;

			if(!is360) ang *= angles / LMath::pi;
			else ang *= angles / (2 * LMath::pi);

			int bin = (int)ang;
			diffData[bin % angles] = (bin + 1 - ang) * mag;
			diffData[(bin + 1) % angles] = (ang - bin) * mag;
		}
		diffData = diffImage.GetData();
        for(j = 0; j < height; j++) for(k = 0; k < width * angles; k++, diffData++) (*diffData) += ((k >= angles) ? *(diffData - angles) : 0) + ((j > 0) ? *(diffData - width * angles) : 0) - (((k >= angles) && (j > 0)) ? *(diffData - angles * width - angles) : 0);

		for(r = 0; r < sizeCount; r++)
		{
			double *respData = siftResponse.GetData();
			for(j = 0; j < subHeight; j++) for(k = 0; k < subWidth; k++, respData += angles * windowNumber * windowNumber)
			{
				for(l = 0; l < windowNumber; l++) for(m = 0; m < windowNumber; m++)
				{
					int x = subSample * k + m * windowSize[r] - (windowSize[r] * windowNumber - diff) / 2, y = subSample * j + l * windowSize[r] - (windowSize[r] * windowNumber - diff) / 2;
					x = (x < 0) ? 0 : ((x > width - 1 - windowSize[r]) ? width - 1 - windowSize[r] : x);
					y = (y < 0) ? 0 : ((y > height - 1 - windowSize[r]) ? height - 1 - windowSize[r] : y);

					for(n = 0; n < angles; n++) *(respData + l * (angles * windowNumber) + m * angles + n) = diffImage(x, y, n) + diffImage(x + windowSize[r], y + windowSize[r], n) - diffImage(x, y + windowSize[r], n) - diffImage(x + windowSize[r], y, n);
				}
				double sum = 0;
				for(l = 0; l < angles * windowNumber * windowNumber; l++) sum += *(respData + l) * *(respData + l);
				if(sum > 0)
				{
					sum = sqrt(sum);
					for(l = 0; l < angles * windowNumber * windowNumber; l++) *(respData + l) /= sum;
				}
			}
			memcpy(dataTo, siftResponse.GetData(), subWidth * subHeight * angles * windowNumber * windowNumber * sizeof(double));
			dataTo += subWidth * subHeight * angles * windowNumber * windowNumber;
		}
	}
	printf("Calculating K-Means clusters..\n");

	clustering->Cluster(siftData, totalSize * sizeCount);
	delete[] siftData;

	SaveTraining();
}

void LSiftFeature::Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName)
{
	int i, j, k, l, m, n, r;
	int width = labImage.GetWidth();
	int height = labImage.GetHeight();

	LImage<double> siftResponse(width, height, angles * windowNumber * windowNumber);

	LImage<double> diffImage(width, height, angles);
	memset(diffImage.GetData(), 0, width * height * angles * sizeof(double));

	double *diffData = diffImage.GetData();
	for(j = 0; j < height; j++) for(k = 0; k < width; k++, diffData += angles)
	{
		double dx, dy, mag, ang;
		dx = labImage((k == width - 1) ? k : k + 1, j, 0) - labImage((k == 0) ? 0 : k - diff, j, 0);
		dy = labImage(k, (j == height - 1) ? j : j + 1, 0) - labImage(k, (j == 0) ? 0 : j - diff, 0);

		mag = sqrt(dx * dx + dy * dy);
		ang = atan2(dy, dx);

		if(ang < 0) ang += 2 * LMath::pi;

		if(!is360) ang *= angles / LMath::pi;
		else ang *= angles / (2 * LMath::pi);

		int bin = (int)ang;
		diffData[bin % angles] = (bin + 1 - ang) * mag;
		diffData[(bin + 1) % angles] = (ang - bin) * mag;
	}
	diffData = diffImage.GetData();
    for(j = 0; j < height; j++) for(k = 0; k < width * angles; k++, diffData++) (*diffData) += ((k >= angles) ? *(diffData - angles) : 0) + ((j > 0) ? *(diffData - width * angles) : 0) - (((k >= angles) && (j > 0)) ? *(diffData - angles * width - angles) : 0);

	image.SetResolution(width, height, sizeCount);

	for(r = 0; r < sizeCount; r++)
	{
		double *respData = siftResponse.GetData();
		for(j = 0; j < height; j++) for(k = 0; k < width; k++, respData += angles * windowNumber * windowNumber)
		{
			for(l = 0; l < windowNumber; l++) for(m = 0; m < windowNumber; m++)
			{
				int x = k + m * windowSize[r] - (windowSize[r] * windowNumber - diff) / 2, y = j + l * windowSize[r] - (windowSize[r] * windowNumber - diff) / 2;
				x = (x < 0) ? 0 : ((x > width - 1 - windowSize[r]) ? width - 1 - windowSize[r] : x);
				y = (y < 0) ? 0 : ((y > height - 1 - windowSize[r]) ? height - 1 - windowSize[r] : y);
				for(n = 0; n < angles; n++) *(respData + l * (angles * windowNumber) + m * angles + n) = diffImage(x, y, n) + diffImage(x + windowSize[r], y + windowSize[r], n) - diffImage(x, y + windowSize[r], n) - diffImage(x + windowSize[r], y, n);
			}
			double sum = 0;
			for(l = 0; l < angles * windowNumber * windowNumber; l++) sum += *(respData + l) * *(respData + l);
			if(sum > 0)
			{
				sum = sqrt(sum);
				for(l = 0; l < angles * windowNumber * windowNumber; l++) *(respData + l) /= sum;
			}
		}
		for(i = 0; i < height; i++) for(j = 0; j < width; j++) image(j, i, r) = clustering->NearestNeighbour(siftResponse(j, i));
	}
}

void LSiftFeature::LoadTraining()
{
	clustering->LoadTraining();
}

void LSiftFeature::SaveTraining()
{
	clustering->SaveTraining();
}

int LSiftFeature::GetBuckets()
{
	return(clustering->GetClusters());
}

LLbpFeature::LLbpFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setWindowSize, int setSubSample, int setNumberOfClusters, double setKMeansMaxChange, int setPointsPerKDTreeCluster) : LDenseFeature(setDataset, setTrainFolder, setTrainFile, setEvalFolder, setEvalExtension)
{
	subSample = setSubSample;
	numberOfClusters = setNumberOfClusters, kMeansMaxChange = setKMeansMaxChange, pointsPerKDTreeCluster = setPointsPerKDTreeCluster;
	windowSize = setWindowSize;
	clustering = new LKMeansClustering<double>(setTrainFolder, setTrainFile, windowSize * windowSize, numberOfClusters, kMeansMaxChange, pointsPerKDTreeCluster, 0);
}

LLbpFeature::~LLbpFeature()
{
	delete clustering;
}

void LLbpFeature::Train(LList<char *> &trainImageFiles)
{
	int fileSum = trainImageFiles.GetCount(), totalSize = 0, i, j, k, l, m;
	double *lbpData, *dataTo;

	printf("Counting total lbp response size..\n");

	for(i = 0; i < trainImageFiles.GetCount(); i++)
	{
		char *fileName;
		fileName = GetFileName(dataset->imageFolder, trainImageFiles[i], dataset->imageExtension);
		LRgbImage rgbImage(fileName);
		delete[] fileName;

		int subWidth = (rgbImage.GetWidth() + subSample - 1) / subSample;
		int subHeight = (rgbImage.GetHeight() + subSample - 1) / subSample;
		totalSize += subWidth * subHeight;
	}
	lbpData = new double[totalSize * windowSize * windowSize];
	dataTo = lbpData;

	printf("Calculating sampled lbp responses..\n");

	for(i = 0; i < trainImageFiles.GetCount(); i++)
	{
		char *fileName;
		fileName = GetFileName(dataset->imageFolder, trainImageFiles[i], dataset->imageExtension);
		LLabImage labImage(fileName);
		delete[] fileName;

		int width = labImage.GetWidth();
		int height = labImage.GetHeight();
		int subWidth = (width + subSample - 1) / subSample;
		int subHeight = (height + subSample - 1) / subSample;

		for(j = 0; j < subHeight; j++) for(k = 0; k < subWidth; k++, dataTo += windowSize * windowSize)
		{
			double thresVal = labImage(k * subSample, j * subSample, 0);
			int fromX = k * subSample - ((windowSize - 1) >> 1);
			int fromY = j * subSample - ((windowSize - 1) >> 1);
			for(l = 0; l < windowSize; l++) for(m = 0; m < windowSize; m++)
			{
				int x = (fromX + m >= 0) ? ((fromX + m < width) ? fromX + m : width - 1) : 0;
				int y = (fromY + l >= 0) ? ((fromY + l < height) ? fromY + l : height - 1) : 0;
				dataTo[l * windowSize + m] = (labImage(x, y, 0) > thresVal) ? 1 : 0;
			}
		}
	}
	printf("Calculating K-Means clusters..\n");

	clustering->Cluster(lbpData, totalSize);
	delete[] lbpData;

	SaveTraining();
}

void LLbpFeature::Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName)
{
	int i, j, l, m;
	int width = labImage.GetWidth();
	int height = labImage.GetHeight();

	double *resp = new double[windowSize * windowSize];

	image.SetResolution(width, height, 1);
	for(i = 0; i < height; i++) for(j = 0; j < width; j++)
	{
		double thresVal = labImage(j, i, 0);
		int fromX = j - ((windowSize - 1) >> 1);
		int fromY = i - ((windowSize - 1) >> 1);
		for(l = 0; l < windowSize; l++) for(m = 0; m < windowSize; m++)
		{
			int x = (fromX + m >= 0) ? ((fromX + m < width) ? fromX + m : width - 1) : 0;
			int y = (fromY + l >= 0) ? ((fromY + l < height) ? fromY + l : height - 1) : 0;
			resp[l * windowSize + m] = (labImage(x, y, 0) > thresVal) ? 1 : 0;
		}
		image(j, i, 0) = clustering->NearestNeighbour(resp);
	}
	if(resp != NULL) delete[] resp;
}

void LLbpFeature::LoadTraining()
{
	clustering->LoadTraining();
}

void LLbpFeature::SaveTraining()
{
	clustering->SaveTraining();
}

int LLbpFeature::GetBuckets()
{
	return(clustering->GetClusters());
}

LColourSiftFeature::LColourSiftFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setWindowSize, int setWindowNumber, int set360, int setAngles, int setSubSample, int setNumberOfClusters, double setKMeansMaxChange, int setPointsPerKDTreeCluster, int setSymetric) : LDenseFeature(setDataset, setTrainFolder, setTrainFile, setEvalFolder, setEvalExtension)
{
	subSample = setSubSample, is360 = set360, angles = setAngles;
	numberOfClusters = setNumberOfClusters, kMeansMaxChange = setKMeansMaxChange, pointsPerKDTreeCluster = setPointsPerKDTreeCluster;
	sizeCount = 1;
	windowSize = new int[1];
	windowSize[0] = setWindowSize;
	windowNumber = setWindowNumber;
	clustering = new LKMeansClustering<double>(setTrainFolder, setTrainFile, 3 * angles * windowNumber * windowNumber, numberOfClusters, kMeansMaxChange, pointsPerKDTreeCluster);
	diff = !setSymetric;
}

LColourSiftFeature::LColourSiftFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setSizeCount, int *setWindowSize, int setWindowNumber, int set360, int setAngles, int setSubSample, int setNumberOfClusters, double setKMeansMaxChange, int setPointsPerKDTreeCluster, int setSymetric) : LDenseFeature(setDataset, setTrainFolder, setTrainFile, setEvalFolder, setEvalExtension)
{
	subSample = setSubSample, is360 = set360, angles = setAngles;
	numberOfClusters = setNumberOfClusters, kMeansMaxChange = setKMeansMaxChange, pointsPerKDTreeCluster = setPointsPerKDTreeCluster;
	sizeCount = setSizeCount;
	windowSize = new int[sizeCount];
	memcpy(windowSize, setWindowSize, sizeCount * sizeof(int));
	windowNumber = setWindowNumber;
	clustering = new LKMeansClustering<double>(setTrainFolder, setTrainFile, 3 * angles * windowNumber * windowNumber, numberOfClusters, kMeansMaxChange, pointsPerKDTreeCluster);
	diff = !setSymetric;
}

LColourSiftFeature::~LColourSiftFeature()
{
	delete clustering;
	delete[] windowSize;
}

void LColourSiftFeature::Train(LList<char *> &trainImageFiles)
{
	int fileSum = trainImageFiles.GetCount(), totalSize = 0, i, j, k, l, m, n, o, r;
	double *siftData, *dataTo;

	printf("Counting total colour sift response size..\n");

	for(i = 0; i < trainImageFiles.GetCount(); i++)
	{
		char *fileName;
		fileName = GetFileName(dataset->imageFolder, trainImageFiles[i], dataset->imageExtension);
		LRgbImage rgbImage(fileName);
		delete[] fileName;

		int subWidth = (rgbImage.GetWidth() + subSample - 1) / subSample;
		int subHeight = (rgbImage.GetHeight() + subSample - 1) / subSample;
		totalSize += subWidth * subHeight;
	}
	siftData = new double[totalSize * 3 * angles * windowNumber * windowNumber * sizeCount];
	dataTo = siftData;

	printf("Calculating sampled colour sift responses..\n");

	for(i = 0; i < trainImageFiles.GetCount(); i++)
	{
		char *fileName;
		fileName = GetFileName(dataset->imageFolder, trainImageFiles[i], dataset->imageExtension);
		LRgbImage rgbImage(fileName);
		delete[] fileName;

		int width = rgbImage.GetWidth();
		int height = rgbImage.GetHeight();
		int subWidth = (width + subSample - 1) / subSample;
		int subHeight = (height + subSample - 1) / subSample;

		LImage<double> colourImage(width, height, 3);

		unsigned char *rgbData = rgbImage.GetData();
		double *colourData = colourImage.GetData();

		for(j = 0; j < width * height; j++, rgbData += 3, colourData += 3)
		{
			colourData[0] = ((int)rgbData[0] - (int)rgbData[1]) / sqrt(2.0);
			colourData[1] = ((int)rgbData[0] + (int)rgbData[1] - 2 * (int)rgbData[2]) / sqrt(6.0);
			colourData[2] = ((int)rgbData[0] + (int)rgbData[1] + (int)rgbData[2]) / sqrt(3.0);
		}
		LImage<double> siftResponse(subWidth, subHeight, 3 * angles * windowNumber * windowNumber);

		LImage<double> diffImage(width, height, 3 * angles);
		memset(diffImage.GetData(), 0, width * height * 3 * angles * sizeof(double));

		double *diffData = diffImage.GetData();
		for(j = 0; j < height; j++) for(k = 0; k < width; k++, diffData += 3 * angles) for(o = 0; o < 3; o++)
		{
			double dx, dy, mag, ang;
			dx = colourImage((k == width - 1) ? k : k + 1, j, o) - colourImage((k == 0) ? 0 : k - diff, j, o);
			dy = colourImage(k, (j == height - 1) ? j : j + 1, o) - colourImage(k, (j == 0) ? 0 : j - diff, o);

			mag = sqrt(dx * dx + dy * dy);
			ang = atan2(dy, dx);
			if(ang < 0) ang += 2 * LMath::pi;

			if(!is360) ang *= angles / LMath::pi;
			else ang *= angles / (2 * LMath::pi);

			int bin = (int)ang;
			diffData[o * angles + (bin % angles)] = (bin + 1 - ang) * mag;
			diffData[o * angles + ((bin + 1) % angles)] = (ang - bin) * mag;
		}
		diffData = diffImage.GetData();
        for(j = 0; j < height; j++) for(k = 0; k < width * 3 * angles; k++, diffData++) (*diffData) += ((k >= 3 * angles) ? *(diffData - 3 * angles) : 0) + ((j > 0) ? *(diffData - 3 * width * angles) : 0) - (((k >= 3 * angles) && (j > 0)) ? *(diffData - 3 * angles * width - 3 * angles) : 0);

		for(r = 0; r < sizeCount; r++)
		{
			double *respData = siftResponse.GetData();

			for(j = 0; j < subHeight; j++) for(k = 0; k < subWidth; k++, respData += 3 * angles * windowNumber * windowNumber)
			{
				for(l = 0; l < windowNumber; l++) for(m = 0; m < windowNumber; m++)
				{
					int x = subSample * k + m * windowSize[r] - (windowSize[r] * windowNumber - diff) / 2, y = subSample * j + l * windowSize[r] - (windowSize[r] * windowNumber - diff) / 2;
					x = (x < 0) ? 0 : ((x > width - 1 - windowSize[r]) ? width - 1 - windowSize[r] : x);
					y = (y < 0) ? 0 : ((y > height - 1 - windowSize[r]) ? height - 1 - windowSize[r] : y);

					for(n = 0; n < 3 * angles; n++) *(respData + l * (3 * angles * windowNumber) + m * 3 * angles + n) = diffImage(x, y, n) + diffImage(x + windowSize[r], y + windowSize[r], n) - diffImage(x, y + windowSize[r], n) - diffImage(x + windowSize[r], y, n);
				}
				double sum = 0;
				for(l = 0; l < 3 * angles * windowNumber * windowNumber; l++) sum += *(respData + l) * *(respData + l);
				if(sum > 0)
				{
					sum = sqrt(sum);
					for(l = 0; l < 3 * angles * windowNumber * windowNumber; l++) *(respData + l) /= sum;
				}
			}
			memcpy(dataTo, siftResponse.GetData(), subWidth * subHeight * 3 * angles * windowNumber * windowNumber * sizeof(double));

			dataTo += subWidth * subHeight * 3 * angles * windowNumber * windowNumber;
		}
	}
	printf("Calculating K-Means clusters..\n");

	clustering->Cluster(siftData, totalSize * sizeCount);
	delete[] siftData;

	SaveTraining();
}

void LColourSiftFeature::Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName)
{
	int i, j, k, l, m, n, o, r;
	int width = labImage.GetWidth();
	int height = labImage.GetHeight();

	LRgbImage rgbImage(labImage);

	LImage<double> colourImage(width, height, 3);

	unsigned char *rgbData = rgbImage.GetData();
	double *colourData = colourImage.GetData();

	for(j = 0; j < width * height; j++, rgbData += 3, colourData += 3)
	{
		colourData[0] = ((int)rgbData[0] - (int)rgbData[1]) / sqrt(2.0);
		colourData[1] = ((int)rgbData[0] + (int)rgbData[1] - 2 * (int)rgbData[2]) / sqrt(6.0);
		colourData[2] = ((int)rgbData[0] + (int)rgbData[1] + (int)rgbData[2]) / sqrt(3.0);
	}

	LImage<double> siftResponse(width, height, 3 * angles * windowNumber * windowNumber);

	LImage<double> diffImage(width, height, 3 * angles);
	memset(diffImage.GetData(), 0, width * height * 3 * angles * sizeof(double));

	double *diffData = diffImage.GetData();
	for(j = 0; j < height; j++) for(k = 0; k < width; k++, diffData += 3 * angles) for(o = 0; o < 3; o++)
	{
		double dx, dy, mag, ang;
		dx = colourImage((k == width - 1) ? k : k + 1, j, o) - colourImage((k == 0) ? 0 : k - diff, j, o);
		dy = colourImage(k, (j == height - 1) ? j : j + 1, o) - colourImage(k, (j == 0) ? 0 : j - diff, o);

		mag = sqrt(dx * dx + dy * dy);
		ang = atan2(dy, dx);
		if(ang < 0) ang += 2 * LMath::pi;

		if(!is360) ang *= angles / LMath::pi;
		else ang *= angles / (2 * LMath::pi);

		int bin = (int)ang;
		diffData[o * angles + (bin % angles)] = (bin + 1 - ang) * mag;
		diffData[o * angles + ((bin + 1) % angles)] = (ang - bin) * mag;
	}
	diffData = diffImage.GetData();
    for(j = 0; j < height; j++) for(k = 0; k < width * 3 * angles; k++, diffData++) (*diffData) += ((k >= 3 * angles) ? *(diffData - 3 * angles) : 0) + ((j > 0) ? *(diffData - 3 * width * angles) : 0) - (((k >= 3 * angles) && (j > 0)) ? *(diffData - 3 * angles * width - 3 * angles) : 0);

	image.SetResolution(width, height, sizeCount);

	for(r = 0; r < sizeCount; r++)
	{
		double *respData = siftResponse.GetData();

		for(j = 0; j < height; j++) for(k = 0; k < width; k++, respData += 3 * angles * windowNumber * windowNumber)
		{
			for(l = 0; l < windowNumber; l++) for(m = 0; m < windowNumber; m++)
			{
				int x = k + m * windowSize[r] - (windowSize[r] * windowNumber - diff) / 2, y = j + l * windowSize[r] - (windowSize[r] * windowNumber - diff) / 2;
				x = (x < 0) ? 0 : ((x > width - 1 - windowSize[r]) ? width - 1 - windowSize[r] : x);
				y = (y < 0) ? 0 : ((y > height - 1 - windowSize[r]) ? height - 1 - windowSize[r] : y);
				for(n = 0; n < 3 * angles; n++) *(respData + l * (3 * angles * windowNumber) + m * 3 * angles + n) = diffImage(x, y, n) + diffImage(x + windowSize[r], y + windowSize[r], n) - diffImage(x, y + windowSize[r], n) - diffImage(x + windowSize[r], y, n);
			}
			double sum = 0;
			for(l = 0; l < 3 * angles * windowNumber * windowNumber; l++) sum += *(respData + l) * *(respData + l);
			if(sum > 0)
			{
				sum = sqrt(sum);
				for(l = 0; l < 3 * angles * windowNumber * windowNumber; l++) *(respData + l) /= sum;
			}
		}
		for(i = 0; i < height; i++) for(j = 0; j < width; j++) image(j, i, r) = clustering->NearestNeighbour(siftResponse(j, i));
	}
}

void LColourSiftFeature::LoadTraining()
{
	clustering->LoadTraining();
}

void LColourSiftFeature::SaveTraining()
{
	clustering->SaveTraining();
}

int LColourSiftFeature::GetBuckets()
{
	return(clustering->GetClusters());
}

LDummyFeature::LDummyFeature(LDataset *setDataset, const char *setEvalFolder, const char *setEvalExtension, int setNumberOfClusters) : LDenseFeature(setDataset, NULL, NULL, setEvalFolder, setEvalExtension)
{
	numberOfClusters = setNumberOfClusters;
}

int LDummyFeature::GetBuckets()
{
	return(numberOfClusters);
}