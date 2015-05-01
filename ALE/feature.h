#ifndef __feature
#define __feature

#include "std.h"
#include "image.h"
#include "clustering.h"
#include "segmentation.h"

class LFeature
{
	protected :
		LDataset *dataset;
		const char *trainFolder, *trainFile;
	public :
		const char *evalFolder, *evalExtension;

		LFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension);
		virtual ~LFeature() {};
		void Evaluate(LList<char *> &imageFiles, int from = 0, int to = -1);

		virtual void LoadTraining() = 0;
		virtual void SaveTraining() = 0;
		virtual void Train(LList<char *> &trainImageFiles) = 0;
		virtual int GetBuckets() = 0;
		virtual void Evaluate(char *imageFileName) {};
};

class LDenseFeature : public LFeature
{
	public :
		LDenseFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension);

		virtual void Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName) = 0;
		virtual void Evaluate(char *imageFileName);
};

class LTextonFeature : public LDenseFeature
{
	private :
		LList<LFilter2D<double> *> filters;
		int subSample, numberOfClusters, pointsPerKDTreeCluster;
		double filterScale, kMeansMaxChange;
		LKMeansClustering<double> *clustering;

		void CreateFilterList();
	public :
		LTextonFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, double setFilterScale, int setSubSample, int setNumberOfClusters, double setKMeansMaxChange, int setPointsPerKDTreeCluster);

		~LTextonFeature();
		int GetTotalFilterCount();

		void Train(LList<char *> &trainImageFiles);
		void Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName);
		void LoadTraining();
		void SaveTraining();
		int GetBuckets();
};

class LSiftFeature : public LDenseFeature
{
	private :
		int subSample, numberOfClusters, pointsPerKDTreeCluster, *windowSize, windowNumber, is360, angles, diff, sizeCount;
		double kMeansMaxChange;
		LKMeansClustering<double> *clustering;
	public :
		LSiftFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setWindowSize, int setWindowNumber, int set360, int setAngles, int setSubSample = 0, int setNumberOfClusters = 0, double setKMeansMaxChange = 0, int setPointsPerKDTreeCluster = 0, int setSymetric = 1);
		LSiftFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setSizeCount, int *setWindowSize, int setWindowNumber, int set360, int setAngles, int setSubSample = 0, int setNumberOfClusters = 0, double setKMeansMaxChange = 0, int setPointsPerKDTreeCluster = 0, int setSymetric = 1);
		~LSiftFeature();

		void Train(LList<char *> &trainImageFiles);
		void Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName);
		void LoadTraining();
		void SaveTraining();
		int GetBuckets();
};

class LColourSiftFeature : public LDenseFeature
{
	private :
		int subSample, numberOfClusters, pointsPerKDTreeCluster, *windowSize, windowNumber, is360, angles, diff, sizeCount;
		double kMeansMaxChange;
		LKMeansClustering<double> *clustering;
	public :
		LColourSiftFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setWindowSize, int setWindowNumber, int set360, int setAngles, int setSubSample = 0, int setNumberOfClusters = 0, double setKMeansMaxChange = 0, int setPointsPerKDTreeCluster = 0, int setSymetric = 1);
		LColourSiftFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setSizeCount, int *setWindowSize, int setWindowNumber, int set360, int setAngles, int setSubSample = 0, int setNumberOfClusters = 0, double setKMeansMaxChange = 0, int setPointsPerKDTreeCluster = 0, int setSymetric = 1);
		~LColourSiftFeature();

		void Train(LList<char *> &trainImageFiles);
		void Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName);
		void LoadTraining();
		void SaveTraining();
		int GetBuckets();
};

class LLbpFeature : public LDenseFeature
{
	private :
		int subSample, numberOfClusters, pointsPerKDTreeCluster, windowSize;
		double kMeansMaxChange;
		LKMeansClustering<double> *clustering;
	public :
		LLbpFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setWindowSize, int setSubSample, int setNumberOfClusters, double setKMeansMaxChange, int setPointsPerKDTreeCluster);
		~LLbpFeature();

		void Train(LList<char *> &trainImageFiles);
		void Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName);
		void LoadTraining();
		void SaveTraining();
		int GetBuckets();
};

class LLocationFeature : public LDenseFeature
{
	private :
		LLatticeClustering<double> *clustering;
	public :
		LLocationFeature(LDataset *setDataset, const char *setEvalFolder, const char *setEvalExtension, int setLocationBuckets);
		~LLocationFeature();

		void Train(LList<char *> &trainImageFiles) {};
		void Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName);
		void LoadTraining() {};
		void SaveTraining() {};
		int GetBuckets();
};

class LDummyFeature : public LDenseFeature
{
	private :
		int numberOfClusters;
	public :
		LDummyFeature(LDataset *setDataset, const char *setEvalFolder, const char *setEvalExtension, int setNumberOfClusters);

		void Evaluate(char *imageFileName) {};
		void Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName) {};
		void Train(LList<char *> &trainImageFiles) {};
		void LoadTraining() {};
		void SaveTraining() {};
		int GetBuckets();
};

#endif