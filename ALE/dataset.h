#ifndef __dataset
#define __dataset

#include <stdio.h>
#include "std.h"

class LCrf;
class LCrfDomain;
class LCrfLayer;
class LBaseCrfLayer;
class LPnCrfLayer;
class LPreferenceCrfLayer;
class LLabelImage;

class LDataset
{
	private :
		char *GetFolderFileName(const char *imageFile, const char *folder, const char *extension);
		static int SortStr(char *str1, char *str2);
	protected :
		void LoadFolder(const char *folder, const char *extension, LList<char *> &list);
	public :
		LDataset();
		virtual ~LDataset();
		virtual void Init();

		LList<char *> trainImageFiles, testImageFiles, allImageFiles;

		unsigned int seed;
		int classNo, filePermutations, featuresOnline;
		double proportionTrain, proportionTest;

		const char *imageFolder, *imageExtension, *groundTruthFolder, *groundTruthExtension, *trainFolder, *testFolder, *dispTestFolder;
		int optimizeAverage;

		int clusterPointsPerKDTreeCluster;
		double clusterKMeansMaxChange;

		int locationBuckets;
		const char *locationFolder, *locationExtension;

		int textonNumberOfClusters, textonKMeansSubSample;
		double textonFilterBankRescale;
		const char *textonClusteringTrainFile, *textonFolder, *textonExtension;

		int siftKMeansSubSample, siftNumberOfClusters, siftWindowSize, siftWindowNumber, sift360, siftAngles, siftSizeCount, siftSizes[4];
		const char *siftClusteringTrainFile, *siftFolder, *siftExtension;

		int colourSiftKMeansSubSample, colourSiftNumberOfClusters, colourSiftWindowSize, colourSiftWindowNumber, colourSift360, colourSiftAngles, colourSiftSizeCount, colourSiftSizes[4];
		const char *colourSiftClusteringTrainFile, *colourSiftFolder, *colourSiftExtension;

		int lbpSize, lbpKMeansSubSample, lbpNumberOfClusters;
		const char *lbpClusteringFile, *lbpFolder, *lbpExtension;

		int denseNumRoundsBoosting, denseBoostingSubSample, denseNumberOfThetas, denseThetaStart, denseThetaIncrement, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize;
		double denseRandomizationFactor, denseWeight, denseMaxClassRatio;
		const char *denseBoostTrainFile, *denseExtension, *denseFolder;

		double meanShiftXY[4], meanShiftLuv[4];
		int meanShiftMinRegion[4];
		const char *meanShiftFolder[4], *meanShiftExtension;

		double consistencyPrior;

		int statsThetaStart, statsThetaIncrement, statsNumberOfThetas, statsNumberOfBoosts;
		double statsRandomizationFactor, statsAlpha, statsFactor, statsPrior, statsMaxClassRatio;
		const char *statsTrainFile, *statsExtension, *statsFolder;

		double pairwiseLWeight, pairwiseUWeight, pairwiseVWeight, pairwisePrior, pairwiseFactor, pairwiseBeta;
		double cliqueMinLabelRatio, cliqueThresholdRatio, cliqueTruncation;

		int pairwiseSegmentBuckets;
		double pairwiseSegmentPrior, pairwiseSegmentFactor, pairwiseSegmentBeta;

		const char *cooccurenceTrainFile;
		double cooccurenceWeight;

		double disparityUnaryFactor, disparityFilterSigma, disparityMaxDistance, disparityDistanceBeta;
		int disparitySubSample, disparityMaxDelta, disparityClassNo, disparityRangeMoveSize;
		double disparityPairwiseFactor, disparityPairwiseTruncation;
		const char *disparityLeftFolder, *disparityRightFolder, *disparityGroundTruthFolder, *disparityGroundTruthExtension;

		double cameraBaseline, cameraHeight, cameraFocalLength, cameraAspectRatio, cameraWidthOffset, cameraHeightOffset;
		double crossUnaryWeight, crossPairwiseWeight, crossMinHeight, crossMaxHeight, crossThreshold;
		int crossHeightClusters;
		const char *crossTrainFile;

		double kMeansXyLuvRatio[6];
		const char *kMeansFolder[6], *kMeansExtension;
		int kMeansIterations, kMeansMaxDiff, kMeansDistance[6];

		int unaryWeighted;
		double *unaryWeights;

		virtual void RgbToLabel(unsigned char *rgb, unsigned char *label);
		virtual void LabelToRgb(unsigned char *label, unsigned char *rgb);

		virtual void SaveImage(LLabelImage &labelImage, LCrfDomain *domain, char *fileName);
		virtual void SetCRFStructure(LCrf *crf) {};
		virtual int Segmented(char *imageFileName);
		virtual void GetLabelSet(unsigned char *labelset, char *imageFileName);
};

class LSowerbyDataset : public LDataset
{
	// sky      0, grass   1, roadline 2, road    3
	// building 4, sign    5, car      6

	private :
	public :
		LSowerbyDataset();
		void SetCRFStructure(LCrf *crf);
};

class LCorelDataset : public LDataset
{
	// rhino/hippo 0, polarbear 1, water 2, snow    3
	// vegetation  4, ground    5, sky   6

	private :
	public :
		LCorelDataset();
		void SetCRFStructure(LCrf *crf);
};

class LMsrcDataset : public LDataset
{
	// building  0, grass     1, tree     2, cow       3
	// horse     4, sheep     5, sky      6, mountain  7
	// plane     8, water     9, face    10, car      11
	// bike     12, flower   13, sign    14, bird     15
	// book     16, chair    17, road    18, cat      19
	// dog      20, body     21, boat    22

	private :
	protected :
		void RgbToLabel(unsigned char *rgb, unsigned char *label);
		void LabelToRgb(unsigned char *label, unsigned char *rgb);
	public :
		const char *trainFileList, *testFileList;
		void Init();

		LMsrcDataset();
		void SetCRFStructure(LCrf *crf);
};

class LLeuvenDataset : public LDataset
{
	// building 0, tree     1, sky         2, car   3
	// sign     4, road     5, pedestrian  6, fense 7
	// column   8, pavement 9, bicyclist  10,
	private :
		void AddFolder(char *folder, LList<char *> &fileList);
	protected :
		void RgbToLabel(unsigned char *rgb, unsigned char *label);
		void LabelToRgb(unsigned char *label, unsigned char *rgb);
	public :
		void Init();

		LLeuvenDataset();
		void SetCRFStructure(LCrf *crf);

		void DisparityRgbToLabel(unsigned char *rgb, unsigned char *label);
		void DisparityLabelToRgb(unsigned char *label, unsigned char *rgb);
};

class LVOCDataset : public LDataset
{
	// background  0, aeroplane  1, bicycle    2, bird       3
	// boat        4, bottle     5, bus        6, car        7
	// cat         8, chair      9, cow       10, din.table 11
	// dog        12, horse     13, motorbike 14, person    15
	// plant      16, sheep     17, sofa      18, train     19
	// tv-monitor 20

	private :
	protected :
	public :
		const char *trainFileList, *testFileList;
		void Init();

		LVOCDataset();
		void RgbToLabel(unsigned char *rgb, unsigned char *label);
		void LabelToRgb(unsigned char *label, unsigned char *rgb);
		void SaveImage(LLabelImage &labelImage, LCrfDomain *domain, char *fileName);
		void SetCRFStructure(LCrf *crf);
};

class LCamVidDataset : public LDataset
{
	// building 0, tree     1, sky         2, car   3
	// sign     4, road     5, pedestrian  6, fense 7
	// column   8, pavement 9, bicyclist  10,

	private :
		void AddFolder(char *folder, LList<char *> &fileList);
	protected :
		void Init();
	public :
		LCamVidDataset();
		
		void RgbToLabel(unsigned char *rgb, unsigned char *label);
		void LabelToRgb(unsigned char *label, unsigned char *rgb);
		void SetCRFStructure(LCrf *crf);
};

#endif