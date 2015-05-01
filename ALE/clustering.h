#ifndef __clustering
#define __clustering

#include <stdio.h>
#include "std.h"

template <class T>
class LClustering
{
	protected :
		const char *clusterFolder, *clusterFile;
		int bands;
	public :
		LClustering(const char *setClusterFolder, const char *setClusterFile, int setBands);
		virtual ~LClustering() {};

		virtual int NearestNeighbour(T *values) = 0;
		virtual void Cluster(T *data, int size) = 0;
		virtual void LoadTraining() = 0;
		virtual void LoadTraining(FILE *f) = 0;
		virtual void SaveTraining() = 0;
		virtual void SaveTraining(FILE *f) = 0;
		virtual int GetClusters() = 0;
};

template <class T>
class LLatticeClustering : public LClustering<T>
{
	private :
		T *minValues, *maxValues;
		int *buckets;
	public :
		LLatticeClustering(int setBands, T *setMinValues, T *setMaxValues, int setBuckets);
		LLatticeClustering(int setBands, T *setMinValues, T *setMaxValues, int *setBuckets);
		~LLatticeClustering();

		int NearestNeighbour(T *values);
		void Cluster(T *data, int size) {};
		void LoadTraining() {};
		void LoadTraining(FILE *f) {};
		void SaveTraining() {};
		void SaveTraining(FILE *f) {};
		int GetClusters();
};

template <class T>
class LKMeansClustering : public LClustering<T>
{
	private :
		class LCluster
		{
			private :
				int bands;
				double *means;
				double *variances;
			public :
				int count;
				double logMixCoeff, logDetCov;

				LCluster();
				LCluster(int setBands);
				~LCluster();

				void SetBands(int setBands);
				double *GetMeans();
				double *GetVariances();
		};
		class LKdTreeNode
		{
			public :
				int terminal;
				int *indices;
				double splitValue;
				int indiceSize, splitDim;
				LKdTreeNode *left, *right;

				LKdTreeNode();
				LKdTreeNode(int *setIndices, int setIndiceSize);
				~LKdTreeNode();
				void SetAsNonTerminal(int setSplitDim, double setSplitValue, LKdTreeNode *setLeft, LKdTreeNode *setRight);
		};
		class LKdTree
		{
			public :
				LKdTreeNode *root;

				LKdTree();
				LKdTree(T *data, int numberOfClusters, int bands, int pointsPerKDTreeCluster);
				~LKdTree();
				int NearestNeighbour(T *data, int bands, double *values, LKdTreeNode *node, double (*meassure)(double *, double *, int));
		};

		int numberOfClusters, pointsPerKDTreeCluster;
		double kMeansMaxChange;

		double *clusterMeans, *dataMeans, *dataVariances;
		LKdTree *kd;
		double (*meassure)(double *, double *, int);
		int finalClusters, normalize;

		double AssignLabels(T *data, int size, LCluster **tempClusters, int *labels);
	public :
		LKMeansClustering(const char *setClusterFolder, const char *setClusterFile, int setBands, int setNumberOfClusters, double setKMeansMaxChange, int setPointsPerKDTreeCluster, int setNormalize = 1);
		~LKMeansClustering();

		int NearestNeighbour(T *values);
		void Cluster(T *data, int size);
		void FillMeans(double *data);
		void LoadTraining();
		void LoadTraining(FILE *f);
		void SaveTraining();
		void SaveTraining(FILE *f);
		int GetClusters();
		double *GetMeans();
};

#endif