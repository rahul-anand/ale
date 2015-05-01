#include <stdio.h>
#include <string.h>
#include <math.h>
#include "clustering.h"

template class LClustering<double>;
template class LLatticeClustering<double>;
template class LKMeansClustering<double>;

template <class T>
LClustering<T>::LClustering(const char *setClusterFolder, const char *setClusterFile, int setBands)
{
	clusterFolder = setClusterFolder, clusterFile = setClusterFile, bands = setBands;
}

template <class T>
LLatticeClustering<T>::LLatticeClustering(int setBands, T *setMinValues, T *setMaxValues, int setBuckets) : LClustering<T>(NULL, NULL, setBands)
{
	minValues = new T[this->bands];
	maxValues = new T[this->bands];
	buckets = new int[this->bands];
	memcpy(minValues, setMinValues, this->bands * sizeof(T));
	memcpy(maxValues, setMaxValues, this->bands * sizeof(T));
	for(int i = 0; i < this->bands; i++) buckets[i] = setBuckets;
}

template <class T>
LLatticeClustering<T>::LLatticeClustering(int setBands, T *setMinValues, T *setMaxValues, int *setBuckets) : LClustering<T>(NULL, NULL, setBands)
{
	minValues = new T[this->bands];
	maxValues = new T[this->bands];
	buckets = new int[this->bands];
	memcpy(minValues, setMinValues, this->bands * sizeof(T));
	memcpy(maxValues, setMaxValues, this->bands * sizeof(T));
	memcpy(buckets, setBuckets, this->bands * sizeof(int));
}

template <class T>
LLatticeClustering<T>::~LLatticeClustering()
{
	if(minValues != NULL) delete[] minValues;
	if(maxValues != NULL) delete[] maxValues;
	if(buckets != NULL) delete[] buckets;
}

template <class T>
int LLatticeClustering<T>::NearestNeighbour(T *values)
{
	int bucket = 0;
	for(int i = 0; i < this->bands; i++)
	{
		int index = (int)((values[i] - minValues[i]) * buckets[i] / (maxValues[i] - minValues[i]));
		index = (index < 0) ? 0 : ((index >= buckets[i]) ? buckets[i] - 1 : index);
		if(i > 0) bucket *= buckets[i - 1];
		bucket += index;
	}
	return(bucket);
}

template <class T>
int LLatticeClustering<T>::GetClusters()
{
	int total = buckets[0];
	for(int i = 1; i < this->bands; i++) total *= buckets[i];
	return(total);
}

template <class T>
LKMeansClustering<T>::LCluster::LCluster()
{
	this->bands = count = 0;
	means = variances = NULL;
	logMixCoeff = logDetCov = 1;
}

template <class T>
LKMeansClustering<T>::LCluster::LCluster(int setBands)
{
	this->bands = setBands, count = 0;
	means = new double[setBands];
	variances = new double[setBands];
	logMixCoeff = logDetCov = 1;
}

template <class T>
LKMeansClustering<T>::LCluster::~LCluster()
{
	if(means != NULL) delete[] means;
	if(variances != NULL) delete[] variances;
}

template <class T>
void LKMeansClustering<T>::LCluster::SetBands(int setBands)
{
	if(means != NULL) delete[] means;
	if(variances != NULL) delete[] variances;

	this->bands = setBands;
	means = new double[setBands];
	variances = new double[setBands];
}

template <class T>
double *LKMeansClustering<T>::LCluster::GetMeans()
{
	return(means);
}

template <class T>
double *LKMeansClustering<T>::LCluster::GetVariances()
{
	return(variances);
}

template <class T>
LKMeansClustering<T>::LKdTreeNode::LKdTreeNode()
{
	left = right = NULL;
	indices = NULL;
	indiceSize = splitDim = 0;
	splitValue = 0, terminal = 1;
}

template <class T>
LKMeansClustering<T>::LKdTreeNode::LKdTreeNode(int *setIndices, int setIndiceSize)
{
	terminal = 1;
	splitDim = 0;
	splitValue = 0;
	left = right = NULL;

	indiceSize = setIndiceSize;
	indices = new int[indiceSize];
	for(int i = 0; i < indiceSize; i++) indices[i] = setIndices[i];
}

template <class T>
void LKMeansClustering<T>::LKdTreeNode::SetAsNonTerminal(int setSplitDim, double setSplitValue, LKdTreeNode *setLeft, LKdTreeNode *setRight)
{
	indiceSize = 0;
	if(indices != NULL) delete[] indices;
	indices = NULL;

	left = setLeft;
	right = setRight;

	terminal = 0;
	splitValue = setSplitValue;
	splitDim = setSplitDim;
}

template <class T>
LKMeansClustering<T>::LKdTreeNode::~LKdTreeNode()
{
	if(left != NULL) delete left;
	if(right != NULL) delete right;
	if(indices != NULL) delete[] indices;
}

template <class T>
LKMeansClustering<T>::LKdTree::LKdTree()
{
	root = NULL;
}

template <class T>
LKMeansClustering<T>::LKdTree::~LKdTree()
{
	if(root != NULL) delete root;
}

template <class T>
LKMeansClustering<T>::LKdTree::LKdTree(T *data, int numberOfClusters, int bands, int pointsPerKDTreeCluster)
{
	int *indices, i, j;
	LList<LKdTreeNode *> toProcess;

	indices = new int[numberOfClusters];
	for(i = 0; i < numberOfClusters; i++) indices[i] = i;

	root = new LKdTreeNode(indices, numberOfClusters);

	toProcess.Add(root);

    while(toProcess.GetCount() > 0)
	{
		LKdTreeNode *node = toProcess[0];
		toProcess.Delete(0);

        int indSize = node->indiceSize;
        
		if(indSize > pointsPerKDTreeCluster)
        {
			double *variances, *x, *x2;

			variances = new double[bands];
			x = new double[bands];
			x2 = new double[bands];

			for(j = 0; j < bands; j++) x[j] = 0, x2[j] = 0;

			for(i = 0; i < indSize; i++) for(j = 0; j < bands; j++)
			{
				x[j] += data[node->indices[i] * bands + j];
				x2[j] += data[node->indices[i] * bands + j] * data[node->indices[i] * bands + j];
			}
			for(j = 0; j < bands; j++) variances[j] = x2[j] / indSize - x[j] * x[j] / (indSize * indSize);
			
			int hDim = 0;
            for(j = 1; j < bands; j++) if (variances[j] > variances[hDim]) hDim = j;

			if(variances[hDim] != 0)
			{
				double split = x[hDim] / indSize;

				LList<int> leftIndices, rightIndices;
				LKdTreeNode *left, *right;

				for (int i = 0; i < indSize; i++)
				{
					if(data[node->indices[i] * bands + hDim] < split) leftIndices.Add(node->indices[i]);
                    else rightIndices.Add(node->indices[i]);
				}

				left = new LKdTreeNode(leftIndices.GetArray(), leftIndices.GetCount());
                right = new LKdTreeNode(rightIndices.GetArray(), rightIndices.GetCount());

                node->SetAsNonTerminal(hDim, split, left, right);
				toProcess.Add(left);
                toProcess.Add(right);
            }
			delete[] variances;
			delete[] x;
			delete[] x2;
		}
	}
}

template <class T>
int LKMeansClustering<T>::LKdTree::NearestNeighbour(T *data, int bands, double *values, LKdTreeNode *node, double (*meassure)(double *, double *, int))
{
	int bestIndex, bestIndex2;

	if(node->terminal)
    {
		bestIndex = node->indices[0];
		double bestDistance = meassure(data + bestIndex * bands, values, bands);

		for(int i = 1; i < node->indiceSize; i++)
        {
			int index = node->indices[i];
            double dist = meassure(data + index * bands, values, bands);

            if (dist < bestDistance)
            {
				bestIndex = index;
                bestDistance = dist;
			}
		}
		return(bestIndex);
	}
	else if (values[node->splitDim] < node->splitValue)
	{
		bestIndex = NearestNeighbour(data, bands, values, node->left, meassure);
        
		double dist = meassure(data + bestIndex * bands, values, bands);
		if((node->splitValue - values[node->splitDim]) * (node->splitValue - values[node->splitDim]) <= dist)
		{
			bestIndex2 = NearestNeighbour(data, bands, values, node->right, meassure);
			double dist2 = meassure(data + bestIndex2 * bands, values, bands);
			return((dist < dist2) ? bestIndex : bestIndex2);
		}
		else return(bestIndex);
	}
	else
	{
		bestIndex = NearestNeighbour(data, bands, values, node->right, meassure);
        
		double dist = meassure(data + bestIndex * bands, values, bands);
		if((node->splitValue - values[node->splitDim]) * (node->splitValue - values[node->splitDim]) <= dist)
		{
			bestIndex2 = NearestNeighbour(data, bands, values, node->left, meassure);
			double dist2 = meassure(data + bestIndex2 * bands, values, bands);
			return((dist < dist2) ? bestIndex : bestIndex2);
		}
		else return(bestIndex);
	}
}

template <class T>
LKMeansClustering<T>::LKMeansClustering(const char *setClusterFolder, const char *setClusterFile, int setBands, int setNumberOfClusters, double setKMeansMaxChange, int setPointsPerKDTreeCluster, int setNormalize) : LClustering<T>(setClusterFolder, setClusterFile, setBands)
{
	finalClusters = 0;
	clusterMeans = NULL;
	kd = NULL;
	dataMeans = dataVariances = NULL;
	meassure = LMath::SquareEuclidianDistance;
	numberOfClusters = setNumberOfClusters, kMeansMaxChange = setKMeansMaxChange, pointsPerKDTreeCluster = setPointsPerKDTreeCluster;
	normalize = setNormalize;
}

template <class T>
void LKMeansClustering<T>::Cluster(T *data, int size)
{
	int *labels, i, j, randIndex, converged = 0;
	double *means, *variances, currentChange = 2 * kMeansMaxChange;
	LCluster **tempClusters;
	LList<LCluster *> clusters;
	double *data1, *data2;

	if(kd != NULL) delete kd;
	if(clusterMeans != NULL) delete[] clusterMeans;
	if(dataMeans != NULL) delete[] dataMeans;
	if(dataVariances != NULL) delete[] dataVariances;

	dataMeans = new double[this->bands];
	dataVariances = new double[this->bands];

	data1 = new double[this->bands];
	data2 = new double[this->bands];

	memset(data1, 0, this->bands * sizeof(double));
	memset(data2, 0, this->bands * sizeof(double));

	for(i = 0; i < size; i++) for(j = 0; j < this->bands; j++) data1[j] += data[i * this->bands + j], data2[j] += data[i * this->bands + j] * data[i * this->bands + j];
	for(j = 0; j < this->bands; j++)  dataMeans[j] = data1[j] / size;
	for(j = 0; j < this->bands; j++)  dataVariances[j] = sqrt(data2[j] / size - (data1[j] / size) * (data1[j] / size));

	delete[] data1;
	delete[] data2;

	if(normalize) for(i = 0; i < size; i++) for(j = 0; j < this->bands; j++) data[i * this->bands + j] = (data[i * this->bands + j] - dataMeans[j]) / dataVariances[j];

	labels = new int[size];
	tempClusters = new LCluster *[numberOfClusters];
	for(i = 0; i < numberOfClusters; i++) tempClusters[i] = new LCluster();

	for(i = 0; i < size; i++) labels[i] = 0;

	for(i = 0; i < numberOfClusters; i++)
	{
		tempClusters[i]->SetBands(this->bands);

		randIndex = LMath::RandomInt(size);
		means = tempClusters[i]->GetMeans();
		variances = tempClusters[i]->GetVariances();

		for(j = 0; j < this->bands; j++)
		{
			means[j] = data[randIndex * this->bands + j];
			variances[j] = 1.0;
		}
		tempClusters[i]->logMixCoeff = log((double)1.0 / numberOfClusters);
        tempClusters[i]->count = 10;
	}
	AssignLabels(data, size, tempClusters, labels);

	while(currentChange >= kMeansMaxChange)
	{
		for(i = 0; i < numberOfClusters; i++)
	    {
			tempClusters[i]->count = 0;
			double *means = tempClusters[i]->GetMeans();
		    for(j = 0; j < this->bands; j++) means[j] = 0;
		}

        for(i = 0; i < size; i++)
        {
	        int label = labels[i];
			double *means = tempClusters[label]->GetMeans();

			for(j = 0; j < this->bands; j++) means[j] += data[i * this->bands + j];
	        tempClusters[label]->count++;
        }

		for(i = 0; i < numberOfClusters; i++) if(tempClusters[i]->count > 0)
		{
			double *means = tempClusters[i]->GetMeans();
			double *variances = tempClusters[i]->GetVariances();

			for(j = 0; j < this->bands; j++)
			{
				means[j] /= tempClusters[i]->count;
				variances[j] = 0;
			}
        }

	    for(i = 0; i < size; i++)
	    {
			int label = labels[i];
			double *means = tempClusters[label]->GetMeans();
			double *variances = tempClusters[label]->GetVariances();

		    for(j = 0; j < this->bands; j++)
		    {
				double diff = data[i * this->bands + j] - means[j];
                variances[j] += diff * diff;
			}
	    }

		for(i = 0; i < numberOfClusters; i++) if(tempClusters[i]->count > 0)
		{
            double detCov = 1.0;
			double *variances = tempClusters[i]->GetVariances();

			for (j = 0; j < this->bands; j++)
            {
                if (variances[j] == 0.0)
				{
					variances[j] = LMath::positiveInfinity;
					detCov = 0;
				}
				else
				{
					variances[j] = tempClusters[i]->count / variances[j];
					detCov /= variances[j];
				}
            }

            double mixCoeff = tempClusters[i]->count / (double) numberOfClusters;

            tempClusters[i]->logDetCov = (detCov > 0) ? log(detCov) : LMath::negativeInfinity;
            tempClusters[i]->logMixCoeff = (mixCoeff > 0) ? log(mixCoeff) : LMath::negativeInfinity;
		}
		currentChange = AssignLabels(data, size, tempClusters, labels);
	}
	delete[] labels;

	for(i = 0; i < numberOfClusters; i++) if(tempClusters[i]->count > 0)  clusters.Add(tempClusters[i]);

	finalClusters = clusters.GetCount();
	clusterMeans = new double [finalClusters * this->bands];

	for(i = 0; i < clusters.GetCount(); i++) memcpy(clusterMeans + i * this->bands, clusters[i]->GetMeans(), this->bands * sizeof(double));
	for(i = 0; i < numberOfClusters; i++) delete tempClusters[i];
	delete[] tempClusters;

	kd = new LKdTree(clusterMeans, finalClusters, this->bands, pointsPerKDTreeCluster);
}

template <class T>
LKMeansClustering<T>::~LKMeansClustering()
{
	if(kd != NULL) delete kd;
	if(clusterMeans != NULL) delete[] clusterMeans;
	if(dataMeans != NULL) delete[] dataMeans;
	if(dataVariances != NULL) delete[] dataVariances;
}

template <class T>
double LKMeansClustering<T>::AssignLabels(T *data, int size, LCluster **tempClusters, int *labels)
{
	double *accMatrix, *sVector, *uVector;
	int i, j;

	accMatrix = new double[numberOfClusters * numberOfClusters];
	sVector = new double[numberOfClusters];
	uVector = new double[size];

    for(i = 0; i < size; i++) uVector[i] = LMath::positiveInfinity;

	for(i = 0; i < numberOfClusters; i++) if(tempClusters[i]->count > 0)
	{
		for(j = 0; j < numberOfClusters; j++)
		{
			if(tempClusters[j]->count > 0) accMatrix[i + j * numberOfClusters] = meassure(tempClusters[i]->GetMeans(), tempClusters[j]->GetMeans(), this->bands);
		}
	}

	for(i = 0; i < numberOfClusters; i++) if(tempClusters[i]->count > 0)
	{
		sVector[i] = LMath::positiveInfinity;
		for(j = 0; j < numberOfClusters; j++)
		{
			if((i != j) && (tempClusters[j]->count > 0))
			{
				double d = accMatrix[i + j * numberOfClusters];
				if(d < sVector[i]) sVector[i] = d;
			}
        }
        sVector[i] /= 2.0;
	}

	int changedCount = 0;
	        
	for(i = 0; i < size; i++)
	{
		int oldLabel = labels[i];
		if(uVector[i] > sVector[oldLabel])
		{
			double currentDist = meassure(tempClusters[oldLabel]->GetMeans(), data + i * this->bands, this->bands);
		    uVector[i] = currentDist;

	        if (currentDist > sVector[oldLabel])
			{
		        int changed = 0;

				for(int j = 0; j < numberOfClusters; j++)
		        {
					if((j != oldLabel) && (tempClusters[j]->count > 0) && (accMatrix[oldLabel + j * numberOfClusters] < 2 * currentDist))
					{
						double dist = meassure(tempClusters[j]->GetMeans(), data + i * this->bands, this->bands);
						if (dist < currentDist)
						{
							labels[i] = j;
							currentDist = dist;
							uVector[i] = currentDist;
							changed = 1;
						}
					}
				}
				if (changed) changedCount++;
			}
		}
	}

	delete[] accMatrix;
	delete[] sVector;
	delete[] uVector;

	return(changedCount / (double)size);
}

template <class T>
int LKMeansClustering<T>::NearestNeighbour(T *values)
{
	if(normalize) for(int j = 0; j < this->bands; j++) values[j] = (values[j] - dataMeans[j]) / dataVariances[j];
	return(kd->NearestNeighbour(clusterMeans, this->bands, values, kd->root, meassure));
}

template <class T>
void LKMeansClustering<T>::FillMeans(double *data)
{
	for(int i = 0; i < finalClusters; i++) for(int j = 0; j < this->bands; j++) data[i * this->bands + j] = clusterMeans[i * this->bands + j] * dataVariances[j] + dataMeans[j];
}

template <class T>
double *LKMeansClustering<T>::GetMeans()
{
	return(clusterMeans);
}

template <class T>
int LKMeansClustering<T>::GetClusters()
{
	return(finalClusters);
}

template <class T>
void LKMeansClustering<T>::SaveTraining()
{
	char *fileName;
	FILE *f;

	fileName = new char[strlen(this->clusterFolder) + strlen(this->clusterFile) + 1];
	sprintf(fileName, "%s%s", this->clusterFolder, this->clusterFile);

	f = fopen(fileName, "wb");
	SaveTraining(f);
	
	fclose(f);
	delete[] fileName;
}

template <class T>
void LKMeansClustering<T>::SaveTraining(FILE *f)
{
	fwrite(&finalClusters, sizeof(int), 1, f);
	fwrite(&this->bands, sizeof(int), 1, f);
	fwrite(clusterMeans, sizeof(double), finalClusters * this->bands, f);

	LList<LKdTreeNode *> toProcess;
	toProcess.Add(kd->root);

	while(toProcess.GetCount() > 0)
	{
		LKdTreeNode *node;
		node = toProcess[0];
		toProcess.Delete(0);

		fwrite(&node->terminal, sizeof(int), 1, f);
		fwrite(&node->indiceSize, sizeof(int), 1, f);
		if(node->indiceSize > 0) fwrite(node->indices, sizeof(int), node->indiceSize, f);

		fwrite(&node->splitDim, sizeof(int), 1, f);
		fwrite(&node->splitValue, sizeof(double), 1, f);

		if(!node->terminal)
		{
			toProcess.Add(node->left);
			toProcess.Add(node->right);
		}
	}
	fwrite(dataMeans, sizeof(double), this->bands, f);
	fwrite(dataVariances, sizeof(double), this->bands, f);
}

template <class T>
void LKMeansClustering<T>::LoadTraining()
{
	char *fileName;
	FILE *f;

	fileName = new char[strlen(this->clusterFolder) + strlen(this->clusterFile) + 1];
	sprintf(fileName, "%s%s", this->clusterFolder, this->clusterFile);

	f = fopen(fileName, "rb");
	LoadTraining(f);

	fclose(f);
	delete[] fileName;
}

template <class T>
void LKMeansClustering<T>::LoadTraining(FILE *f)
{
	meassure = LMath::SquareEuclidianDistance;

	fread(&finalClusters, sizeof(int), 1, f);
	fread(&this->bands, sizeof(int), 1, f);

	if(clusterMeans != NULL) delete[] clusterMeans;
	clusterMeans = new double[finalClusters * this->bands];
	fread(clusterMeans, sizeof(double), finalClusters * this->bands, f);

	kd = new LKdTree();
	kd->root = new LKdTreeNode();

	LList<LKdTreeNode *> toProcess;
	toProcess.Add(kd->root);

	while(toProcess.GetCount() > 0)
	{
		LKdTreeNode *node;
		node = toProcess[0];
		toProcess.Delete(0);

		fread(&node->terminal, sizeof(int), 1, f);
		fread(&node->indiceSize, sizeof(int), 1, f);
		if(node->indiceSize > 0)
		{
			node->indices = new int[node->indiceSize];
			fread(node->indices, sizeof(int), node->indiceSize, f);
		}

		fread(&node->splitDim, sizeof(int), 1, f);
		fread(&node->splitValue, sizeof(double), 1, f);

		if(!node->terminal)
		{
			node->left = new LKdTreeNode();
			toProcess.Add(node->left);

			node->right = new LKdTreeNode();
			toProcess.Add(node->right);
		}
	}
	if(dataMeans != NULL) delete[] dataMeans;
	if(dataVariances != NULL) delete[] dataVariances;

	dataMeans = new double[this->bands];
	dataVariances = new double[this->bands];

	fread(dataMeans, sizeof(double), this->bands, f);
	fread(dataVariances, sizeof(double), this->bands, f);
}

