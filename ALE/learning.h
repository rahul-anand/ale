#ifndef __learning
#define __learning

#include <stdio.h>
#include "std.h"
#include "potential.h"

class LLearning
{
	private :
	protected :
		const char *trainFolder, *trainFile;
		int classNo;
	public :
		LLearning(const char *setTrainFolder, const char *setTrainFile, int setClassNo);
		virtual ~LLearning() {};

		virtual void Train(double *setClassWeights, int setNumExamples, int *setTargets, int setNumFeatures, double **setWeights) {};
		virtual void Evaluate(double *costs, int total, int setNumFeatures) {};
		virtual void SaveTraining() {};
		virtual void LoadTraining() {};
		virtual int SwapTrainData() { return(0); }
		virtual int SwapEvalData() { return(0); }
};

template <class T>
class LBoostWeakLearner
{
	private :
		int classNo;
	public :
		int index;
		T theta;
		double error, a, b, *k;
		unsigned int sharingSet;

		LBoostWeakLearner();
		LBoostWeakLearner(int setClassNo);
		~LBoostWeakLearner();
		void CopyFrom(LBoostWeakLearner<T> &wl);
};    

template <class T>
class LBoosting : public LLearning
{
	public :
	private :
		int numExamples, *targets, numFeatures;
		double **weights, randomizationFactor;
		int numberOfThetas, numRoundsBoosting;
		T *theta;
		double *classWeights;

		void CalculateWeightSum(double *weightSum, double *weightSumTarget, double *k);
		void CalculateWeightSumPos(double **weightSumPos, double **weightSumTargetPos, T *featureValues);
		void UpdateWeights(LBoostWeakLearner<T> &wl, int index);
		void OptimiseSharing(LBoostWeakLearner<T> &wl, double *weightSum, double *weightSumTarget, double *k, double **weightSumPos, double **weightSumTargetPos);
		void OptimiseWeakLearner(LBoostWeakLearner<T> &wl, unsigned int n, double *weightSum, double *weightSumTarget, double *k, double **weightSumPos, double **weightSumTargetPos);

		LList<LBoostWeakLearner<T> *> weakLearners;
		T *(LPotential::*getTrainValues)(int, int);
		T *(LPotential::*getEvalValues)(int);
		LPotential *potential;
	protected :
	public :
		LBoosting(const char *setTrainFolder, const char *setTrainFile, int setClassNo, int setNumRoundsBoosting, T thetaStart, T thetaIncrement, int setNumberOfThetas, double setRandomizationFactor, LPotential *setPotential, T *(LPotential::*setGetTrainValues)(int, int), T *(LPotential::*setGetEvalValues)(int));
		~LBoosting();
		void Train(double *setClassWeights, int setNumExamples, int *setTargets, int setNumFeatures, double **setWeights);
		void Evaluate(double *costs, int total, int setNumFeatures);

		void SaveTraining();
		void LoadTraining();
		void FindBestFeature(double *weightSum, double *weightSumTarget, double *k, double **weightSumPos, double **weightSumTargetPos, int featureStart, int featureEnd, LBoostWeakLearner<T> *minimum, int core);
		int SwapTrainData() { return(1); }
		int SwapEvalData() { return(1); }
};

template <class T>
class LLinearSVM : public LLearning
{
	public :
	private :
		int numExamples, *targets, numFeatures;
		double **weights;
		T *featureValues;
		double *classWeights;

		double lambda, scale;
		int epochs, skip;
		double **w;
		double *bias;
		double bscale;

		T *(LPotential::*getTrainValues)(int, int);
		T *(LPotential::*getEvalValues)(int);
		LPotential *potential;
	protected :
	public :
		LLinearSVM(const char *setTrainFolder, const char *setTrainFile, int setClassNo, LPotential *setPotential, T *(LPotential::*setGetTrainValues)(int, int), T *(LPotential::*setGetEvalValues)(int), double setLambda, int setEpochs, int setSkip, double setBScale);
		~LLinearSVM();
		void Train(double *setClassWeights, int setNumExamples, int *setTargets, int setNumFeatures, double **setWeights);
		void TrainClass(int classIndex, int core);
		void Evaluate(double *costs, int total, int setNumFeatures);

		void SaveTraining();
		void LoadTraining();
};

template <class T>
class LApproxKernel
{
	public :
		virtual double K(double omega) = 0;
		void Map(T *x1, double *x2, int dim, int approxCount, double L);
};

template <class T>
class LHellingerApproxKernel : public LApproxKernel<T>
{
	public :
		double K(double omega);
};

template <class T>
class LChi2ApproxKernel : public LApproxKernel<T>
{
	public :
		double K(double omega);
};

template <class T>
class LIntersectionApproxKernel : public LApproxKernel<T>
{
	public :
		double K(double omega);
};

template <class T>
class LJensenShannonApproxKernel : public LApproxKernel<T>
{
	public :
		double K(double omega);
};

template <class T>
class LApproxKernelSVM : public LLearning
{
	public :
	private :
		int numExamples, *targets, numFeatures;
		double **weights;
		T *featureValues;
		double *classWeights;
		LApproxKernel<T> *kernel;

		double lambda, bscale;
		int epochs, skip, approxCount;
		double **w, L;
		double *bias;

		T *(LPotential::*getTrainValues)(int, int);
		T *(LPotential::*getEvalValues)(int);
		LPotential *potential;
	protected :
	public :
		LApproxKernelSVM(const char *setTrainFolder, const char *setTrainFile, int setClassNo, LPotential *setPotential, T *(LPotential::*setGetTrainValues)(int, int), T *(LPotential::*setGetEvalValues)(int), double setLambda, int setEpochs, int setSkip, double setScale, LApproxKernel<T> *setKernel, double setL, int setApproxCount);
		~LApproxKernelSVM();
		void Train(double *setClassWeights, int setNumExamples, int *setTargets, int setNumFeatures, double **setWeights);
		void Evaluate(double *costs, int total, int setNumFeatures);
		void TrainClass(int classIndex, int core);

		void SaveTraining();
		void LoadTraining();
};

template <class T>
class LRandomTree
{
	private :
		int classNo, depth;
	public :
		int *index;
		T *theta;
		double *prob;

		LRandomTree();
		LRandomTree(int setClassNo, int setDepth);
		~LRandomTree();
		void CopyFrom(LRandomTree<T> &rt);
};    

template <class T>
class LRandomForest : public LLearning
{
	public :
	private :
		int numExamples, *targets, numFeatures;
		double **weights, randomizationFactor;
		int numberOfThetas, numTrees, depth;
		T *theta;
		double *classWeights, dataRatio;

		LList<LRandomTree<T> *> randomTrees;
		T *(LPotential::*getTrainValues)(int, int);
		T *(LPotential::*getEvalValues)(int);
		T *(LPotential::*getTrainValuesSubset)(int, int *, int, int);
		T (LPotential::*getOneEvalValue)(int, int);
		LPotential *potential;
	protected :
	public :
		LRandomForest(const char *setTrainFolder, const char *setTrainFile, int setClassNo, int setNumTrees, int setDepth, T thetaStart, T thetaIncrement, int setNumberOfThetas, double setRandomizationFactor, LPotential *setPotential, T *(LPotential::*setGetTrainValues)(int, int), T *(LPotential::*setGetEvalValues)(int), double setDataRatio);
		LRandomForest(const char *setTrainFolder, const char *setTrainFile, int setClassNo, int setNumTrees, int setDepth, T thetaStart, T thetaIncrement, int setNumberOfThetas, double setRandomizationFactor, LPotential *setPotential, T *(LPotential::*setGetTrainValuesSubset)(int, int *, int, int), T (LPotential::*setGetOneEvalValue)(int, int), double setDataRatio);
		~LRandomForest();
		void Train(double *setClassWeights, int setNumExamples, int *setTargets, int setNumFeatures, double **setWeights);
		void Evaluate(double *costs, int total, int setNumFeatures);

		void SaveTraining();
		void LoadTraining();
		void FindBestFeature(int *indexes, int indexCount, int featureStart, int featureEnd, int *feature, T *thetaBest, double *minEntropy, int core);
		int SwapTrainData() { return(1); }
};

#endif
