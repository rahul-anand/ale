#ifndef __potential
#define __potential

#include <stdio.h>
#include "feature.h"
#include "dataset.h"
#include "graph.h"

class LLearning;

class LPotential
{
	protected :
		int classNo;
		const char *trainFolder, *trainFile, *evalFolder, *evalExtension;
	public :
		LDataset *dataset;
		LCrf *crf;
		LCrfDomain *domain;
		int nodeOffset, edgeOffset;

		LPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setClassNo);
		virtual ~LPotential() {};

		void Evaluate(LList<char *> &imageFiles, int from = 0, int to = -1);

		virtual int GetNodeCount();
		virtual int GetEdgeCount();

		virtual void Train(LList<char *> &trainImageFiles) {};
		virtual void SaveTraining() {};
		virtual void LoadTraining() {};
		virtual void Evaluate(char *imageFileName) {};
		virtual void Initialize(LLabImage &labImage, char *imageFileName) {};
		virtual void UnInitialize() {};
		virtual void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes) {};
		virtual double GetCost(LCrfDomain *costDomain) { return(0); }
};

class LUnaryPixelPotential : public LPotential
{
	protected :
		double unaryFactor;
		int width, height, unaryCount;
		double *unaryCosts;
		LBaseCrfLayer *layer;
	public :
		LUnaryPixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setUnaryFactor);
		~LUnaryPixelPotential();
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		void SetLabels();
		virtual double GetCost(LCrfDomain *costDomain);
};

class LDisparityUnaryPixelPotential : public LUnaryPixelPotential
{
	protected :
		double filterSigma, maxDistance, distanceBeta;
		int subSample, maxDelta;
		const char *leftFolder, *rightFolder;
	public :
		LDisparityUnaryPixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo, double setUnaryFactor, double setFilterSigma, double setMaxDistance, double setDistanceBeta, int setSubSample, int setMaxDelta, const char *setLeftFolder, const char *setRightFolder);
		~LDisparityUnaryPixelPotential();

		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();
};

class LDenseUnaryPixelPotential : public LUnaryPixelPotential
{
	protected :
		struct LShape
		{
			int x, y, width, height;
		};
		int thetaIncrement, thetaStart, numberOfThetas;

	private :
		LIntegralImage ***integralImages;
		int minimumRectangleSize, maximumRectangleSize;
		int subSample, *buckets, fileSum, **featureValues, numberOfShapes, *integralBuckets, *integralPoints;
		double **weights, *classWeights;
		int *targets, totalFeatures;
		int **validPointsX, **validPointsY, *pointsSum, numExamples;
		LShape *shapes;
		double maxClassRatio;
		LList<LFeature *> features;

		int CalculateShapeFilterResponse(int index, int bucket, LShape *shape, int pointX, int pointY);
		void InitTrainData(LList<char *> &trainImageFiles);
		void InitTrainData(LLabImage *labImages, LLabelImage *groundTruth);
		void UnInitTrainData();
		void InitEvalData(char *evalImageFile, int *width, int *height);
		void InitEvalData(LLabImage &labImage);
		void UnInitEvalData();
		void CalculateIntegralImages(LLabImage &labImage, LIntegralImage ***integralImage, int subSample, int *width, int *height, char *imageFileName);
	public :
		LLearning *learning;

		LDenseUnaryPixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setUnaryFactor, int setSubSample, int setNumberOfShapes, int setMinimumRectangleSize, int setMaximumRectangleSize, double setMaxClassRatio);
		~LDenseUnaryPixelPotential();

		void AddFeature(LDenseFeature *feature);

		void Train(LList<char *> &trainImageFiles);
		void Train(LLabImage *labImages, LLabelImage *groundTruth, int count);

		void SaveTraining();
		void LoadTraining();
		void Evaluate(char *imageFileName);

		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();

		int *GetTrainBoostingValues(int index, int core);
		int *GetEvalBoostingValues(int index);
		int *GetTrainForestValues(int index, int *pixelIndexes, int count, int core);
		int GetEvalForestValue(int index, int pixelIndex);
		int GetLength(int index);
		int GetSize(int index);
};

class LHeightUnaryPixelPotential : public LUnaryPixelPotential
{
	protected :
		double cameraBaseline, cameraHeight, cameraFocalLength, cameraAspectRatio, cameraWidthOffset, cameraHeightOffset;
		double minHeight, maxHeight, threshold;
		int numberOfClusters, disparityClassNo;
		int *heightCounts, *totals, subSample;
		LCrfDomain *objDomain, *dispDomain;
		LBaseCrfLayer *dispLayer, *objLayer;
		int first;
	public :
		LHeightUnaryPixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setObjDomain, LBaseCrfLayer *setObjLayer, LCrfDomain *setDispDomain, LBaseCrfLayer *setDispLayer, const char *setTrainFolder, const char *setTrainFile, int setClassNo, double setUnaryFactor, int setDisparityClassNo, double setCameraBaseline, double setCameraHeight, double setCameraFocalLength, double setCameraAspectRatio, double setCameraWidthOffset, double setCameraHeightOffset, double setMinHeight, double setMaxHeight, int setNumberOfClusters, double setThreshold, int setSubSample);
		~LHeightUnaryPixelPotential();
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);

		void Initialize(LLabImage &labImage, char *imageFileName);
		void Train(LList<char *> &trainImageFiles);
		void SaveTraining();
		void LoadTraining();
		void UnInitialize();
		double GetCost(LCrfDomain *costDomain);
};

class LPairwisePixelPotential : public LPotential
{
	protected :
		LBaseCrfLayer *layer;
	public :
		LPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo);
};

class LPottsPairwisePixelPotential : public LPairwisePixelPotential
{
	protected :
		double *pairwiseCosts;
		int pairwiseCount, *pairwiseIndexes;
	public :
		LPottsPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo);
		~LPottsPairwisePixelPotential();
		int GetEdgeCount();
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		double GetCost(LCrfDomain *costDomain);
};

class LEightNeighbourPottsPairwisePixelPotential : public LPottsPairwisePixelPotential
{
	protected :
		double pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight;
		double PairwiseDiff(double *lab1, double *lab2, double distance);
	public :
		LEightNeighbourPottsPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo, double setPairwisePrior, double setPairwiseFactor, double setPairwiseBeta, double setPairwiseLWeight, double setPairwiseUWeight, double setPairwiseVWeight);
		void Train(LList<char *> &trainImageFiles) {};
		void SaveTraining() {};
		void LoadTraining() {};
		void Evaluate(LList<char *> &imageFiles) {};
		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();
};

class LLinearTruncatedPairwisePixelPotential : public LPairwisePixelPotential
{
	protected :
		double *pairwiseCosts;
		int pairwiseCount, *pairwiseIndexes;
		double pairwiseFactor;
		double truncation;
	public :
		LLinearTruncatedPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo, double setPairwiseFactor, double setTruncation);
		~LLinearTruncatedPairwisePixelPotential();
		int GetEdgeCount();
		int GetNodeCount();
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		double GetCost(LCrfDomain *costDomain);
};

class LDisparityLinearTruncatedPairwisePixelPotential : public LLinearTruncatedPairwisePixelPotential
{
	protected :
	public :
		LDisparityLinearTruncatedPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo, double setPairwiseFactor, double setTruncation);
		void Train(LList<char *> &trainImageFiles) {};
		void SaveTraining() {};
		void LoadTraining() {};
		void Evaluate(LList<char *> &imageFiles) {};
		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();
};

class LJointPairwisePixelPotential : public LPairwisePixelPotential
{
	protected :
		double *pairwiseCosts;
		int pairwiseCount, *pairwiseIndexes;
		int first;

		double *objCosts, disparityFactor, crossFactor, truncation;
		double pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight;
		int disparityClassNo;
		double PairwiseDiff(double *lab1, double *lab2, double distance);

		LCrfDomain *objDomain, *dispDomain;
		LBaseCrfLayer *dispLayer, *objLayer;
	public :
		LJointPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setObjDomain, LBaseCrfLayer *setObjLayer, LCrfDomain *setDispDomain, LBaseCrfLayer *setDispLayer, int setClassNo, int setDisparityClassNo, double setPairwisePrior, double setPairwiseFactor, double setPairwiseBeta, double setPairwiseLWeight, double setPairwiseUWeight, double setPairwiseVWeight, double setDisparityFactor, double setTruncation, double setCrossFactor);
		~LJointPairwisePixelPotential();

		void Train(LList<char *> &trainImageFiles) {};
		void SaveTraining() {};
		void LoadTraining() {};
		void Evaluate(LList<char *> &imageFiles) {};
		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();
		int GetEdgeCount();
		int GetNodeCount();
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		double GetCost(LCrfDomain *costDomain);
};


class LUnarySegmentPotential : public LPotential
{
	protected :
		double consistencyPrior, segmentFactor;
		LList<LPnCrfLayer *> layers;
	public :
		double *unaryCosts;

		LUnarySegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setConsistencyPrior, double setSegmentFactor);
		~LUnarySegmentPotential();
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		void AddLayer(LPnCrfLayer *layer);
		double GetCost(LCrfDomain *costDomain);
};

class LConsistencyUnarySegmentPotential : public LUnarySegmentPotential
{
	protected :
	public :
		LConsistencyUnarySegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, int setClassNo, double setConsistencyPrior);

		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();

		void Train(LList<char *> &trainImageFiles) {};
		void SaveTraining() {};
		void LoadTraining() {};
};

class LStatsUnarySegmentPotential : public LUnarySegmentPotential
{
	protected :
		double *clusterProbabilities, minLabelRatio, kMeansMaxChange, alpha, maxClassRatio;
		int colourStatsClusters, pointsPerKdCluster, *buckets, *buckets2;

		int numExamples, totalFeatures, *targets, totalSegments, width, height;
		double **weights, **featureValues;
		int numberOfThetas;
		double *classWeights, **evalData;
		LSegmentation2D *segmentation;
		LList<LDenseFeature *> features;
		int neighbour;
	public :
		LLearning *learning;

		LStatsUnarySegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setConsistencyPrior, double setSegmentFactor, double setMinLabelRatio, double setAlpha, double setMaxClassRatio, int setNeighbour = 0, LSegmentation2D *setSegmentation = NULL);
		~LStatsUnarySegmentPotential();

		void Train(LList<char *> &trainImageFiles);
		void SaveTraining();
		void LoadTraining();
		void Evaluate(char *imageFileName);
		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();

		void AddFeature(LDenseFeature *feature);

		double *GetTrainBoostingValues(int index, int core);
		double *GetEvalBoostingValues(int index);
		double *GetTrainSVMValues(int index, int core);
		double *GetEvalSVMValues(int index);
};

class LPairwiseSegmentPotential : public LPotential
{
	protected :
		LPnCrfLayer *layer;
	public :
		LPairwiseSegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LPnCrfLayer *setLayer, int setClassNo);
		void Train(LList<char *> &trainImageFiles) {};
		void SaveTraining() {};
		void LoadTraining() {};
};

class LPottsPairwiseSegmentPotential : public LPairwiseSegmentPotential
{
	protected :
		double *pairwiseCosts;
		int pairwiseCount, *pairwiseIndexes;
	public :
		LPottsPairwiseSegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LPnCrfLayer *setLayer, int setClassNo);
		~LPottsPairwiseSegmentPotential();
		int GetEdgeCount();
		int GetNodeCount();
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		double GetCost(LCrfDomain *costDomain);
};

class LHistogramPottsPairwiseSegmentPotential : public LPottsPairwiseSegmentPotential
{
	protected :
		double pairwisePrior, pairwiseFactor, pairwiseBeta;
		int buckets;
		double PairwiseDistance(double *h1, double *h2);
	public :
		LHistogramPottsPairwiseSegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LPnCrfLayer *setLayer, int setClassNo, double setPairwisePrior, double setPairwiseFactor, double setPairwiseBeta, int setBuckets);
		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();
};

class LUnaryImagePotential : public LPotential
{
	protected :
		double *unaryImageFactor;
		LPreferenceCrfLayer *layer;
		double *unaryCosts;
	public :
		LUnaryImagePotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LPreferenceCrfLayer *setLayer, const char *setTrainFolder, const char *setTrainFile, int setClassNo, double *setUnaryImageFactor);
		~LUnaryImagePotential();
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		int GetEdgeCount();
		double GetCost(LCrfDomain *costDomain);
};

class LPairwiseImagePotential : public LPotential
{
	protected :
		double pairwiseImageFactor, *pairwiseCosts;
		LPreferenceCrfLayer *layer;
	public :
		LPairwiseImagePotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LPreferenceCrfLayer *setLayer, const char *setTrainFolder, const char *setTrainFile, int setClassNo, double setpairwiseImageFactor);
		~LPairwiseImagePotential();
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		int GetEdgeCount();
		double GetCost(LCrfDomain *costDomain);
};

class LCooccurencePairwiseImagePotential : public LPairwiseImagePotential
{
	protected :
		int *classOccurence, *classCooccurence, total;
	public :
		LCooccurencePairwiseImagePotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LPreferenceCrfLayer *setLayer, const char *setTrainFolder, const char *setTrainFile, int setClassNo, double setPairwiseImageFactor);
		~LCooccurencePairwiseImagePotential();

		void Train(LList<char *> &trainImageFiles);
		void SaveTraining();
		void LoadTraining();

		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();
};

class LDummyUnaryPixelPotential : public LUnaryPixelPotential
{
	protected :
	public :
		LDummyUnaryPixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setUnaryFactor);

		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();
};

class LDummyUnarySegmentPotential : public LUnarySegmentPotential
{
	protected :
		double alpha;
	public :
		LDummyUnarySegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setSegmentFactor, double setAlpha);

		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();
};

class LDummyDetectorPotential : public LUnarySegmentPotential
{
	protected :
		double detectorThreshold;
	public :
		LDummyDetectorPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LPnCrfLayer *setLayer, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setSegmentFactor, double setDetectorThreshold);

		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();
};

#endif