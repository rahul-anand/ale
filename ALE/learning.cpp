#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "learning.h"

template class LBoostWeakLearner<int>;
template class LBoostWeakLearner<double>;
template class LBoosting<int>;
template class LBoosting<double>;
template class LLinearSVM<int>;
template class LLinearSVM<double>;
template class LRandomTree<int>;
template class LRandomTree<double>;
template class LRandomForest<int>;
template class LRandomForest<double>;
template class LApproxKernel<double>;
template class LChi2ApproxKernel<double>;
template class LIntersectionApproxKernel<double>;
template class LJensenShannonApproxKernel<double>;
template class LHellingerApproxKernel<double>;
template class LApproxKernelSVM<double>;

LLearning::LLearning(const char *setTrainFolder, const char *setTrainFile, int setClassNo)
{
	trainFolder = setTrainFolder, trainFile = setTrainFile, classNo = setClassNo;
}

template <class T>
LBoostWeakLearner<T>::LBoostWeakLearner()
{
	k = NULL;
}

template <class T>
LBoostWeakLearner<T>::LBoostWeakLearner(int setClassNo)
{
	classNo = setClassNo;
	k = new double[classNo];
	memset(k, 0, classNo * sizeof(double));
}

template <class T>
LBoostWeakLearner<T>::~LBoostWeakLearner()
{
	if(k != NULL) delete[] k;
}

template <class T>
void LBoostWeakLearner<T>::CopyFrom(LBoostWeakLearner<T> &wl)
{
	index = wl.index, theta = wl.theta;
	error = wl.error, a = wl.a, b = wl.b;
	memcpy(k, wl.k, classNo * sizeof(double));
	sharingSet = wl.sharingSet;
}

template <class T>
LBoosting<T>::LBoosting(const char *setTrainFolder, const char *setTrainFile, int setClassNo, int setNumRoundsBoosting, T thetaStart, T thetaIncrement, int setNumberOfThetas, double setRandomizationFactor, LPotential *setPotential, T *(LPotential::*setGetTrainValues)(int, int), T *(LPotential::*setGetEvalValues)(int)) : LLearning(setTrainFolder, setTrainFile, setClassNo)
{
	int i;
	numberOfThetas = setNumberOfThetas;
	numRoundsBoosting = setNumRoundsBoosting;

	getTrainValues = setGetTrainValues;
	getEvalValues = setGetEvalValues;
	potential = setPotential;

	theta = new T[numberOfThetas];
	theta[0] = thetaStart;
	if(numberOfThetas > 0) theta[1] = thetaStart + thetaIncrement;
	for(i = 2; i < numberOfThetas; i++) theta[i] = theta[i - 1] + theta[i - 2];
	randomizationFactor = setRandomizationFactor;
}

template <class T>
LBoosting<T>::~LBoosting()
{
	int i;
	if(theta != NULL) delete[] theta;
	for(i = 0; i < weakLearners.GetCount(); i++) delete weakLearners[i];
	weakLearners.Clear();
}


template <class T>
void LBoosting<T>::SaveTraining()
{
	char *fileName;
	FILE *f;

	fileName = new char[strlen(trainFolder) + strlen(trainFile) + 1];
	sprintf(fileName, "%s%s", trainFolder, trainFile);
	f = fopen(fileName, "wb");

	int weakCount = weakLearners.GetCount();
	fwrite(&weakCount, sizeof(int), 1, f);

	for(int i = 0; i < weakLearners.GetCount(); i++)
	{
		LBoostWeakLearner<T> *weak = weakLearners[i];

		fwrite(&weak->index, sizeof(int), 1, f);
		fwrite(&weak->theta, sizeof(T), 1, f);
		fwrite(&weak->a, sizeof(double), 1, f);
		fwrite(&weak->b, sizeof(double), 1, f);
		fwrite(&weak->sharingSet, sizeof(unsigned int), 1, f);
		fwrite(weak->k, sizeof(double), classNo, f);
	}

	fclose(f);
	delete[] fileName;
}

template <class T>
void LBoosting<T>::LoadTraining()
{
	char *fileName;
	FILE *f;
	int i;

	fileName = new char[strlen(trainFolder) + strlen(trainFile) + 1];
	sprintf(fileName, "%s%s", trainFolder, trainFile);
	f = fopen(fileName, "rb");

	for(i = 0; i < weakLearners.GetCount(); i++) delete weakLearners[i];
	weakLearners.Clear();

	int weakCount;
	fread(&weakCount, sizeof(int), 1, f);

	for(i = 0; i < weakCount; i++)
	{
		LBoostWeakLearner<T> *weak = new LBoostWeakLearner<T>(classNo);

		fread(&weak->index, sizeof(int), 1, f);
		fread(&weak->theta, sizeof(T), 1, f);
		fread(&weak->a, sizeof(double), 1, f);
		fread(&weak->b, sizeof(double), 1, f);
		fread(&weak->sharingSet, sizeof(unsigned int), 1, f);
		fread(weak->k, sizeof(double), classNo, f);

		weakLearners.Add(weak);
	}
	fclose(f);
	delete[] fileName;
}

template <class T>
void LBoosting<T>::CalculateWeightSum(double *weightSum, double *weightSumTarget, double *k)
{
    for(int i = 0; i < classNo; i++)
    {
        weightSum[i] = weightSumTarget[i] = 0;
        for (int j = 0; j < numExamples; j++) if(weights[j][i] > 0)
        {
            weightSum[i] += classWeights[targets[j]] * weights[j][i];
            weightSumTarget[i] += classWeights[targets[j]] * weights[j][i] * ((targets[j] == i) ? 1 : -1);
        }
        k[i] = weightSumTarget[i] / weightSum[i];
    }
}

template <class T>
void LBoosting<T>::CalculateWeightSumPos(double **weightSumPos, double **weightSumTargetPos, T *featureValues)
{
	int i, j;
    for(i = 0; i < numberOfThetas; i++) for (j = 0; j < classNo; j++) weightSumPos[i][j] = weightSumTargetPos[i][j] = 0.0;

    for(i = 0; i < numExamples; i++)
    {
		int t = -1;
		while((t < numberOfThetas - 1) && (theta[t + 1] <= featureValues[i])) t++;

        if(t >= 0)
		{
			for (j = 0; j < classNo; j++) if (weights[i][j] > 0)
			{
				weightSumPos[t][j] += classWeights[targets[i]] * weights[i][j];
				weightSumTargetPos[t][j] += classWeights[targets[i]] * weights[i][j] * ((targets[i] == j) ? 1 : -1);
			}
		}
    }
    for(i = numberOfThetas - 2; i >= 0; i--) for (j = 0; j < classNo; j++)
    {
        weightSumPos[i][j] += weightSumPos[i + 1][j];
        weightSumTargetPos[i][j] += weightSumTargetPos[i + 1][j];
    }
}

template <class T>
void LBoosting<T>::UpdateWeights(LBoostWeakLearner<T> &wl, int index)
{
	T *featureValues = (potential->*getTrainValues)(index, 0);

	for (int i = 0; i < numExamples; i++)
    {
        double confidence = (featureValues[i] >= wl.theta) ? (wl.a + wl.b) : wl.b;
        for(int j = 0; j < classNo; j++) if(weights[i][j] > 0) weights[i][j] *= exp(-((targets[i] == j) ? 1 : -1) * ((wl.sharingSet & (1 << j)) ? confidence : wl.k[j]));
    }
}

template <class T>
void LBoosting<T>::OptimiseWeakLearner(LBoostWeakLearner<T> &wl, unsigned int n, double *weightSum, double *weightSumTarget, double *k, double **weightSumPos, double **weightSumTargetPos)
{
	int i, j;
    const double tol = (double)1e-6;

	wl.error = LMath::positiveInfinity;
	wl.sharingSet = n;

    for(i = 0; i < numberOfThetas; i++)
    {
		double p = 0, q = 0, t = 0, u = 0, a, b;

		for(j = 0; j < classNo; j++) if(n & (1 << j)) p += weightSum[j], q += weightSumPos[i][j], t += weightSumTarget[j], u += weightSumTargetPos[i][j];

	    double pq = p - q, tu = t - u;

	    if((fabs(q) >= tol) && (fabs(pq) >= tol))
		{
			b = tu / pq, a = (u / q) - b;

			double Jwse = 0.0;
			for (j = 0; j < classNo; j++)
			{
				if(n & (1 << j)) Jwse += weightSum[j] - 2 * a * weightSumTargetPos[i][j] - 2 * b * weightSumTarget[j] + a * a * weightSumPos[i][j] + b * b * weightSum[j] + 2 * a * b * weightSumPos[i][j];
				else Jwse += weightSum[j] - 2 * k[j] * weightSumTarget[j] + k[j] * k[j] * weightSum[j];
			}

			if((Jwse >= 0) && (Jwse < wl.error))
			{
				wl.error = Jwse, wl.theta = theta[i];
				wl.a = a, wl.b = b;
				for (j = 0; j < classNo; j++) wl.k[j] = (n & (1 << j)) ? 0 : k[j];
			}
		}
    }
}

template <class T>
void LBoosting<T>::OptimiseSharing(LBoostWeakLearner<T> &wl, double *weightSum, double *weightSumTarget, double *k, double **weightSumPos, double **weightSumTargetPos)
{
    unsigned int mask = 0;
	int i, j;
    LBoostWeakLearner<T> **testWL, *optimalWL;
    unsigned int *tempN;

	testWL = new LBoostWeakLearner<T> *[classNo];
	for(i = 0; i < classNo; i++) testWL[i] = new LBoostWeakLearner<T>(classNo);

	tempN = new unsigned int[classNo];
	optimalWL = new LBoostWeakLearner<T>(classNo);

    for(i = 0; i < classNo; i++) testWL[i]->error = LMath::positiveInfinity;

    for (i = 1; i < classNo; i++)
    {
		LList<unsigned int> allowed;

        for(j = 0; j < classNo; j++) if(!(mask & (1 << j))) allowed.Add(mask | (1 << j));
		for(j = 0; j < allowed.GetCount(); j++)
        {
            OptimiseWeakLearner(*optimalWL, allowed[j], weightSum, weightSumTarget, k, weightSumPos, weightSumTargetPos);

            if((optimalWL->error > 0) && (optimalWL->error < testWL[i]->error))
            {
				testWL[i]->CopyFrom(*optimalWL);
                tempN[i] = allowed[j];
            }
        }
        mask = tempN[i];
    }

	wl.CopyFrom(*testWL[1]);
	for(i = 2; i < classNo; i++) if (testWL[i]->error < wl.error) wl.CopyFrom(*testWL[i]);

	delete optimalWL;
	for(i = 0; i < classNo; i++) delete testWL[i];
	delete[] testWL;
	delete[] tempN;
}

template <class T>
void LBoosting<T>::FindBestFeature(double *weightSum, double *weightSumTarget, double *k, double **weightSumPos, double **weightSumTargetPos, int featureStart, int featureEnd, LBoostWeakLearner<T> *minimum, int core)
{
	int feature;
	LBoostWeakLearner<T> optimalSharing(classNo);

	minimum->error = LMath::positiveInfinity;
	for(feature = featureStart; feature <= featureEnd; feature++)
	{
		if(LMath::RandomReal() < randomizationFactor)
		{
			CalculateWeightSumPos(weightSumPos, weightSumTargetPos, (potential->*getTrainValues)(feature, core));
			OptimiseSharing(optimalSharing, weightSum, weightSumTarget, k, weightSumPos, weightSumTargetPos);

			if(optimalSharing.error < minimum->error)
			{
				minimum->CopyFrom(optimalSharing);
				minimum->index = feature;
			}
		}
	}
}

#ifdef MULTITHREAD

template <class T>
struct LBoostingParams
{
	LBoosting<T> *boosting;
	LBoostWeakLearner<T> *minimum;
	int featureFrom, featureTo;
	double *weightSum, *weightSumTarget, *k, **weightSumPos, **weightSumTargetPos;
	unsigned int seed;
	int core;
};

template <class T>
thread_return BoostingThread(void *par)
{
	LBoostingParams<T> *params = (LBoostingParams<T> *)par;
	LMath::SetSeed(params->seed);
	params->boosting->FindBestFeature(params->weightSum, params->weightSumTarget, params->k, params->weightSumPos, params->weightSumTargetPos, params->featureFrom, params->featureTo, params->minimum, params->core);
	return(thread_defoutput);
}
#endif

template <class T>
void LBoosting<T>::Train(double *setClassWeights, int setNumExamples, int *setTargets, int setNumFeatures, double **setWeights)
{
	double *weightSum, *weightSumTarget, *k;
	int i;

	numExamples = setNumExamples, classWeights = setClassWeights, targets = setTargets, numFeatures = setNumFeatures;

	weights = setWeights;
	weightSum = new double[classNo];
    weightSumTarget = new double[classNo];
    k = new double[classNo];

#ifdef MULTITHREAD
	int processors = GetProcessors(), j;

	thread_type *threads;
	LBoostingParams<T> *params;
	LBoostWeakLearner<T> **mins;

	threads = new thread_type[processors];	
	memset(threads, 0, processors * sizeof(thread_type));
	params = new LBoostingParams<T>[processors];
	mins = new LBoostWeakLearner<T> *[processors];
	for(i = 0; i < processors; i++) mins[i] = new LBoostWeakLearner<T>(classNo);

	double ***weightSumPos, ***weightSumTargetPos;

    weightSumPos = new double **[processors];
    for(i = 0; i < processors; i++) weightSumPos[i] = new double *[numberOfThetas];
	for(j = 0; j < processors; j++) for(i = 0; i < numberOfThetas; i++) weightSumPos[j][i] = new double[classNo];

    weightSumTargetPos = new double **[processors];
    for(i = 0; i < processors; i++) weightSumTargetPos[i] = new double *[numberOfThetas];
	for(j = 0; j < processors; j++) for(i = 0; i < numberOfThetas; i++) weightSumTargetPos[j][i] = new double[classNo];

	for(i = 0; i < processors; i++)
	{
		params[i].featureFrom = numFeatures * i / processors;
		params[i].featureTo = numFeatures * (i + 1) / processors - 1;

		params[i].weightSum = weightSum;
		params[i].weightSumTarget = weightSumTarget;
		params[i].k = k;
		params[i].weightSumPos = weightSumPos[i];
		params[i].weightSumTargetPos = weightSumTargetPos[i];
		params[i].minimum = mins[i];
		params[i].boosting = this;
		params[i].core = i;
	}
#else
	double **weightSumPos, **weightSumTargetPos;

	weightSumPos = new double *[numberOfThetas];
	for(i = 0; i < numberOfThetas; i++) weightSumPos[i] = new double[classNo];

	weightSumTargetPos = new double *[numberOfThetas];
	for(i = 0; i < numberOfThetas; i++) weightSumTargetPos[i] = new double[classNo];
#endif

	printf("Boosting..\n");
	for(i = 0; i < numRoundsBoosting; i++)
	{
		printf("Boost no.%d\n", i + 1);
		CalculateWeightSum(weightSum, weightSumTarget, k);

		LBoostWeakLearner<T> *minimum;
		minimum = new LBoostWeakLearner<T>(classNo);

#ifdef MULTITHREAD
		int running = 0;
		for(j = 0; j < processors; j++)
		{
			params[j].seed = LMath::RandomInt();
			threads[j] = NewThread(BoostingThread<T>,  &params[j]);
			running += (threads[j] != 0);
		}
	
		minimum->error = LMath::positiveInfinity;
		while(running)
		{
			for(j = 0; j < processors; j++) if(threads[j])
			{
				if(ThreadFinished(threads[j]))
				{
					CloseThread(&threads[j]);
					running--;
					if(params[j].minimum->error < minimum->error) minimum->CopyFrom(*params[j].minimum);
				}
			}
			Sleep(0);
		}
#else
		FindBestFeature(weightSum, weightSumTarget, k, weightSumPos, weightSumTargetPos, 0, numFeatures - 1, minimum, 0);
#endif
		if(minimum->error < LMath::positiveInfinity)
		{
			UpdateWeights(*minimum, minimum->index);
			weakLearners.Add(minimum);
		}
		else delete minimum;
	}

#ifdef MULTITHREAD
	if(mins != NULL)
	{
		for(i = 0; i < processors; i++) delete mins[i];
		delete[] mins;
	}

	if(weightSumPos != NULL)
	{
		for(j = 0; j < processors; j++) if(weightSumPos[j] != NULL)
		{
			for(i = 0; i < numberOfThetas; i++) if(weightSumPos[j][i] != NULL) delete[] weightSumPos[j][i];
			delete[] weightSumPos[j];
		}
		delete[] weightSumPos;
	}

	if(weightSumTargetPos != NULL)
	{
		for(j = 0; j < processors; j++) if(weightSumTargetPos[j] != NULL)
		{
			for(i = 0; i < numberOfThetas; i++) if(weightSumTargetPos[j][i] != NULL) delete[] weightSumTargetPos[j][i];
			delete[] weightSumTargetPos[j];
		}
		delete[] weightSumTargetPos;
	}
	delete[] threads;
	delete[] params;
#else
	if(weightSumPos != NULL)
	{
		for(i = 0; i < numberOfThetas; i++) if(weightSumPos[i] != NULL) delete[] weightSumPos[i];
		delete[] weightSumPos;
	}
	if(weightSumTargetPos != NULL)
	{
		for(i = 0; i < numberOfThetas; i++) if(weightSumTargetPos[i] != NULL) delete[] weightSumTargetPos[i];
		delete[] weightSumTargetPos;
	}
#endif
	if(weightSum != NULL) delete[] weightSum;
	if(weightSumTarget != NULL) delete[] weightSumTarget;
	if(k != NULL) delete[] k;
}

template <class T>
void LBoosting<T>::Evaluate(double *costs, int total, int setNumFeatures)
{
	int i, j, k;

	numFeatures = setNumFeatures;

	memset(costs, 0, classNo * total * sizeof(double));

	for(j = 0; j < weakLearners.GetCount(); j++)
	{
		T *resp = (potential->*getEvalValues)(weakLearners[j]->index);
		for(i = 0; i < total; i++) 
		{
			double confidenceShared = (resp[i] >= weakLearners[j]->theta) ? (weakLearners[j]->a + weakLearners[j]->b) : weakLearners[j]->b;
			for(k = 0; k < classNo; k++) costs[i * classNo + k] += (weakLearners[j]->sharingSet & (1 << k)) ? confidenceShared : weakLearners[j]->k[k];
		}
	}
}

#ifdef MULTITHREAD

template <class T>
struct LLinearSVMParams
{
	LLinearSVM<T> *svm;
	int classIndex;
	int core;
};

template <class T>
thread_return LinearSVMThread(void *par)
{
	LLinearSVMParams<T> *params = (LLinearSVMParams<T> *)par;
	params->svm->TrainClass(params->classIndex, params->core);
	return(thread_defoutput);
}
#endif

template <class T>
LLinearSVM<T>::LLinearSVM(const char *setTrainFolder, const char *setTrainFile, int setClassNo, LPotential *setPotential, T *(LPotential::*setGetTrainValues)(int, int), T *(LPotential::*setGetEvalValues)(int), double setLambda, int setEpochs, int setSkip, double setBScale) : LLearning(setTrainFolder, setTrainFile, setClassNo)
{
	potential = setPotential;
	getTrainValues = setGetTrainValues;
	getEvalValues = setGetEvalValues;
	lambda = setLambda;
	epochs = setEpochs;
	skip = setSkip;
	w = NULL;
	bias = NULL;
	bscale = setBScale;
}

template <class T>
LLinearSVM<T>::~LLinearSVM()
{
	if(bias != NULL) delete[] bias;
	if(w != NULL)
	{
		for(int i = 0; i < classNo; i++) if(w[i] != NULL) delete[] w[i];
		delete[] w;
	}
}

template <class T>
void LLinearSVM<T>::TrainClass(int classIndex, int core)
{
	int i, k, l, m;

	double count = skip;
	double maxw = 1.0 / sqrt(lambda);
	double typw = sqrt(maxw);
	double eta0 = typw / 1.0;
	double t = 1 / (eta0 * lambda);

	for(l = 0; l < epochs; l++)
	{
        if(!(l % 10)) printf("class %d, epoch %d\n", classIndex, l);
		for(m = 0; m < numExamples; m++)
		{
			i = LMath::RandomInt(numExamples);

			T *x = (potential->*getTrainValues)(i, core);

			double eta = 1.0 / (lambda * t);

			if(weights[i][classIndex] > 0)
			{
				double y = (targets[i] == classIndex) ? 1 : -1;
				double cw = classWeights[targets[i]] * weights[i][classIndex];

				double wx = 0;
				for(k = 0; k < numFeatures; k++) wx += w[classIndex][k] * x[k];

				double z = y * (wx + bias[classIndex]);

				if (z < 1)
				{
					double etd = eta * ((z < 1) ? 1 : 0);

					for(k = 0; k < numFeatures; k++) w[classIndex][k] += etd * y * x[k] * cw;
					bias[classIndex] += etd * y * bscale * cw;
				}
			}

			count--;
			if(count < 0)
			{
				double r = 1 - eta * lambda * skip;
  				if (r < 0.8) r = pow(1 - eta * lambda, skip);

				for(k = 0; k < numFeatures; k++) w[classIndex][k] *= r;
				count += skip;
			}
			t++;
		}
	}
}

template <class T>
void LLinearSVM<T>::Train(double *setClassWeights, int setNumExamples, int *setTargets, int setNumFeatures, double **setWeights)
{
	int i, j;

	numExamples = setNumExamples, classWeights = setClassWeights, targets = setTargets, numFeatures = setNumFeatures;
	weights = setWeights;

	if(bias != NULL) delete[] bias;
	if(w != NULL)
	{
		for(int i = 0; i < classNo; i++) if(w[i] != NULL) delete[] w[i];
		delete[] w;
	}
	bias = new double[classNo];
	memset(bias, 0, classNo * sizeof(double));

	w = new double *[classNo];
	for(i = 0; i < classNo; i++)
	{
		w[i] = new double[numFeatures];
		memset(w[i], 0, numFeatures * sizeof(double));
	}

	double sumw = 0;
	for(i = 0; i < numExamples; i++) for(j = 0; j < classNo; j++) sumw += weights[i][j] * classWeights[targets[i]];
	sumw /= numExamples * classNo;
	for(i = 0; i < classNo; i++) classWeights[i] /= sumw;

#ifdef MULTITHREAD
	int processors = GetProcessors(), running = 0, ind = 0;
	thread_type *threads;
	LLinearSVMParams<T> *params;

	threads = new thread_type[processors];	
	memset(threads, 0, processors * sizeof(thread_type));
	params = new LLinearSVMParams<T>[processors];

	for(i = 0; i < processors; i++) if(ind < classNo)
	{
		params[i].svm = this, params[i].classIndex = ind, params[i].core = i;

		threads[i] = NewThread(LinearSVMThread<T>,  &params[i]);
		if(threads[i] != 0)
		{
			printf("Training class %d..\n", i);
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

				if(ind < classNo)
				{
					params[i].classIndex = ind;
					threads[i] = NewThread(LinearSVMThread<T>,  &params[i]);
					if(threads[i] != 0)
					{
						printf("Training class %d..\n", ind);
						running++;
						ind++;
					}
				}
			}
		}
		Sleep(0);
	}
	delete[] threads;
	delete[] params;
#else
	for(i = 0; i < classNo; i++) 
	{
		printf("Training class %d..\n", i);
		TrainClass(i, 0);
	}
#endif
}

template <class T>
void LLinearSVM<T>::SaveTraining()
{
	char *fileName;
	FILE *f;

	fileName = new char[strlen(trainFolder) + strlen(trainFile) + 1];
	sprintf(fileName, "%s%s", trainFolder, trainFile);
	f = fopen(fileName, "wb");
	fwrite(&numFeatures, sizeof(int), 1, f);

	for(int i = 0; i < classNo; i++) fwrite(w[i], sizeof(double), numFeatures, f);
	fwrite(bias, sizeof(double), classNo, f);

	fclose(f);
	delete[] fileName;
}

template <class T>
void LLinearSVM<T>::LoadTraining()
{
	int i;

	if(bias != NULL) delete[] bias;
	if(w != NULL)
	{
		for(int i = 0; i < classNo; i++) if(w[i] != NULL) delete[] w[i];
		delete[] w;
	}

	char *fileName;
	FILE *f;

	fileName = new char[strlen(trainFolder) + strlen(trainFile) + 1];
	sprintf(fileName, "%s%s", trainFolder, trainFile);
	f = fopen(fileName, "rb");
	fread(&numFeatures, sizeof(int), 1, f);

	bias = new double[classNo];
	w = new double *[classNo];
	for(i = 0; i < classNo; i++) w[i] = new double[numFeatures];
	for(i = 0; i < classNo; i++) fread(w[i], sizeof(double), numFeatures, f);
	fread(bias, sizeof(double), classNo, f);

	fclose(f);
	delete[] fileName;
}

template <class T>
void LLinearSVM<T>::Evaluate(double *costs, int total, int setNumFeatures)
{
	int i, k, l;

	numFeatures = setNumFeatures;

	double *c = costs;
	for(i = 0; i < total; i++, c += classNo)
	{
		T *x = (potential->*getEvalValues)(i);
		for(k = 0; k < classNo; k++)
		{
			c[k] = bias[k];
			for(l = 0; l < numFeatures; l++) c[k] += w[k][l] * x[l];
		}
		i = i + 1 - 1;
	}
}

template <class T>
void LApproxKernel<T>::Map(T *x1, double *x2, int dim, int approxCount, double L)
{
	int i, j;

	for(j = 0; j < dim; j++) x2[j * approxCount] = sqrt(fabs(K(0) * x1[j]));
	for(i = 1; i < approxCount; i++)
	{
		double *data = x2 + i;
		if(i & 1) for(j = 0; j < dim; j++) data[j * approxCount] = sqrt(fabs(2 * x1[j] * L * K(((i + 1) >> 1) * L))) * cos(((i + 1) >> 1) * L * x1[j]);
		else for(j = 0; j < dim; j++) data[j * approxCount] = sqrt(fabs(2 * x1[j] * L * K((i >> 1) * L))) * sin((i >> 1) * L * x1[j]);
	}
};

template <class T>
double LHellingerApproxKernel<T>::K(double omega)
{
	return(!omega);
};

template <class T>
double LChi2ApproxKernel<T>::K(double omega)
{
	return(2 / (exp(LMath::pi * omega) + exp(-LMath::pi * omega)));
};

template <class T>
double LIntersectionApproxKernel<T>::K(double omega)
{
	return(2 / (LMath::pi * (1 + 4 * omega * omega)));
};

template <class T>
double LJensenShannonApproxKernel<T>::K(double omega)
{
	return(4 / (log((double)4.0) * (exp(LMath::pi * omega) + exp(-LMath::pi * omega)) * (LMath::pi * (1 + 4 * omega * omega))));
};

#ifdef MULTITHREAD

template <class T>
struct LApproxKernelSVMParams
{
	LApproxKernelSVM<T> *svm;
	int classIndex;
	int core;
};

template <class T>
thread_return ApproxKernelSVMThread(void *par)
{
	LApproxKernelSVMParams<T> *params = (LApproxKernelSVMParams<T> *)par;
	params->svm->TrainClass(params->classIndex, params->core);
	return(thread_defoutput);
}
#endif

template <class T>
LApproxKernelSVM<T>::LApproxKernelSVM(const char *setTrainFolder, const char *setTrainFile, int setClassNo, LPotential *setPotential, T *(LPotential::*setGetTrainValues)(int, int), T *(LPotential::*setGetEvalValues)(int), double setLambda, int setEpochs, int setSkip, double setBScale, LApproxKernel<T> *setKernel, double setL, int setApproxCount) : LLearning(setTrainFolder, setTrainFile, setClassNo)
{
	potential = setPotential;
	getTrainValues = setGetTrainValues;
	getEvalValues = setGetEvalValues;
	lambda = setLambda;
	epochs = setEpochs;
	skip = setSkip;
	w = NULL;
	bias = NULL;
	bscale = setBScale;
	kernel = setKernel;
	L = setL;
	approxCount = setApproxCount;
}

template <class T>
LApproxKernelSVM<T>::~LApproxKernelSVM()
{
	if(bias != NULL) delete[] bias;
	if(w != NULL)
	{
		for(int i = 0; i < classNo; i++) if(w[i] != NULL) delete[] w[i];
		delete[] w;
	}
	if(kernel) delete kernel;
}

template <class T>
void LApproxKernelSVM<T>::TrainClass(int classIndex, int core)
{
	int i, k, l;

	double *map = new double[approxCount * numFeatures];
	double count = skip;
	double maxw = 1.0 / sqrt(lambda);
	double typw = sqrt(maxw);
	double eta0 = typw / 1.0;
	double t = 1 / (eta0 * lambda);
	
	for(l = 0; l < epochs; l++)
	{
        if(!(l % 10)) printf("class %d, epoch %d\n", classIndex, l);
		for(i = 0; i < numExamples; i++)
		{
			T *x = (potential->*getTrainValues)(i, core);
			kernel->Map(x, map, numFeatures, approxCount, L);

			double eta = 1.0 / (lambda * t);

			if(weights[i][classIndex] > 0)
			{
				double cw = classWeights[targets[i]] * weights[i][classIndex];

				double y = (targets[i] == classIndex) ? 1 : -1;

				double wx = 0;
				for(k = 0; k < numFeatures * approxCount; k++) wx += w[classIndex][k] * map[k];
				double z = y * (wx + bias[classIndex]);

				if (z < 1)
				{
					double etd = eta * ((z < 1) ? 1 : 0);

					for(k = 0; k < numFeatures * approxCount; k++) w[classIndex][k] += etd * y * map[k] * cw;
					bias[classIndex] += etd * y * bscale * cw;
				}
			}

			count--;
			if(count < 0)
			{
				double r = 1 - eta * lambda * skip;
  				if (r < 0.8) r = pow(1 - eta * lambda, skip);

				for(k = 0; k < numFeatures * approxCount; k++) w[classIndex][k] *= r;
				count += skip;
			}
			t++;
		}
	}
	if(map != NULL) delete[] map;
}

template <class T>
void LApproxKernelSVM<T>::Train(double *setClassWeights, int setNumExamples, int *setTargets, int setNumFeatures, double **setWeights)
{
	int i, j;

	numExamples = setNumExamples, classWeights = setClassWeights, targets = setTargets, numFeatures = setNumFeatures;
	weights = setWeights;

	if(bias != NULL) delete[] bias;
	if(w != NULL)
	{
		for(int i = 0; i < classNo; i++) if(w[i] != NULL) delete[] w[i];
		delete[] w;
	}
	bias = new double[classNo];
	memset(bias, 0, classNo * sizeof(double));

	w = new double *[classNo];
	for(i = 0; i < classNo; i++)
	{
		w[i] = new double[numFeatures * approxCount];
		memset(w[i], 0, numFeatures * approxCount * sizeof(double));
	}

	double sumw = 0;
	for(i = 0; i < numExamples; i++) for(j = 0; j < classNo; j++) sumw += weights[i][j] * classWeights[targets[i]];
	sumw /= numExamples * classNo;
	for(i = 0; i < classNo; i++) classWeights[i] /= sumw;

#ifdef MULTITHREAD
	int processors = GetProcessors(), running = 0, ind = 0;
	thread_type *threads;
	LApproxKernelSVMParams<T> *params;

	threads = new thread_type[processors];	
	memset(threads, 0, processors * sizeof(thread_type));
	params = new LApproxKernelSVMParams<T>[processors];

	for(i = 0; i < processors; i++) if(ind < classNo)
	{
		params[i].svm = this, params[i].classIndex = ind, params[i].core = i;

		threads[i] = NewThread(ApproxKernelSVMThread<T>,  &params[i]);
		if(threads[i] != 0)
		{
			printf("Training class %d..\n", i);
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

				if(ind < classNo)
				{
					params[i].classIndex = ind;
					threads[i] = NewThread(ApproxKernelSVMThread<T>,  &params[i]);
					if(threads[i] != 0)
					{
						printf("Training class %d..\n", ind);
						running++;
						ind++;
					}
				}
			}
		}
		Sleep(0);
	}
	delete[] threads;
	delete[] params;
#else
	for(i = 0; i < classNo; i++) 
	{
		printf("Training class %d..\n", i);
		TrainClass(i, 0);
	}
#endif
}

template <class T>
void LApproxKernelSVM<T>::SaveTraining()
{
	char *fileName;
	FILE *f;

	fileName = new char[strlen(trainFolder) + strlen(trainFile) + 1];
	sprintf(fileName, "%s%s", trainFolder, trainFile);
	f = fopen(fileName, "wb");
	fwrite(&numFeatures, sizeof(int), 1, f);

	for(int i = 0; i < classNo; i++) fwrite(w[i], sizeof(double), numFeatures * approxCount, f);
	fwrite(bias, sizeof(double), classNo, f);

	fclose(f);
	delete[] fileName;
}

template <class T>
void LApproxKernelSVM<T>::LoadTraining()
{
	int i;

	if(bias != NULL) delete[] bias;
	if(w != NULL)
	{
		for(int i = 0; i < classNo; i++) if(w[i] != NULL) delete[] w[i];
		delete[] w;
	}

	char *fileName;
	FILE *f;

	fileName = new char[strlen(trainFolder) + strlen(trainFile) + 1];
	sprintf(fileName, "%s%s", trainFolder, trainFile);
	f = fopen(fileName, "rb");
	fread(&numFeatures, sizeof(int), 1, f);

	bias = new double[classNo];
	w = new double *[classNo];
	for(i = 0; i < classNo; i++) w[i] = new double[numFeatures * approxCount];
	for(i = 0; i < classNo; i++) fread(w[i], sizeof(double), numFeatures * approxCount, f);
	fread(bias, sizeof(double), classNo, f);

	fclose(f);
	delete[] fileName;
}

template <class T>
void LApproxKernelSVM<T>::Evaluate(double *costs, int total, int setNumFeatures)
{
	int i, k, l;

	numFeatures = setNumFeatures;

	double *map = new double[approxCount * numFeatures];
	double *c = costs;

	for(i = 0; i < total; i++, c += classNo)
	{
		T *x = (potential->*getEvalValues)(i);
		kernel->Map(x, map, numFeatures, approxCount, L);

		for(k = 0; k < classNo; k++)
		{
			c[k] = bias[k];
			for(l = 0; l < numFeatures * approxCount; l++) c[k] += w[k][l] * map[l];
		}
	}
	if(map != NULL) delete[] map;
}

template <class T>
LRandomTree<T>::LRandomTree()
{
	prob = NULL;
	theta = NULL;
	index = NULL;
}

template <class T>
LRandomTree<T>::LRandomTree(int setClassNo, int setDepth)
{
	classNo = setClassNo;
	depth = setDepth;

	int size = (1 << depth);
	prob = new double[classNo * size];
	theta = new T[size - 1];
	index = new int[size - 1];
	memset(prob, 0, classNo * size * sizeof(double));
}

template <class T>
LRandomTree<T>::~LRandomTree()
{
	if(prob != NULL) delete[] prob;
	if(theta != NULL) delete[] theta;
	if(index != NULL) delete[] index;
}

template <class T>
void LRandomTree<T>::CopyFrom(LRandomTree<T> &rt)
{
	int size = (1 << depth);
	memcpy(prob, rt.prob, classNo * size * sizeof(double));
	memcpy(theta, rt.theta, (size - 1) * sizeof(T));
	memcpy(index, rt.index, (size - 1) * sizeof(int));
}

template <class T>
LRandomForest<T>::LRandomForest(const char *setTrainFolder, const char *setTrainFile, int setClassNo, int setNumTrees, int setDepth, T thetaStart, T thetaIncrement, int setNumberOfThetas, double setRandomizationFactor, LPotential *setPotential, T *(LPotential::*setGetTrainValues)(int, int), T *(LPotential::*setGetEvalValues)(int), double setDataRatio) : LLearning(setTrainFolder, setTrainFile, setClassNo)
{
	int i;
	numberOfThetas = setNumberOfThetas;
	numTrees = setNumTrees;
	dataRatio = setDataRatio;

	getTrainValues = setGetTrainValues;
	getEvalValues = setGetEvalValues;
	getTrainValuesSubset = NULL;
	getOneEvalValue = NULL;
	potential = setPotential;

	theta = new T[numberOfThetas];
	theta[0] = thetaStart;
	if(numberOfThetas > 0) theta[1] = thetaStart + thetaIncrement;
	for(i = 2; i < numberOfThetas; i++) theta[i] = theta[i - 1] + theta[i - 2];
	randomizationFactor = setRandomizationFactor, depth = setDepth;
}

template <class T>
LRandomForest<T>::LRandomForest(const char *setTrainFolder, const char *setTrainFile, int setClassNo, int setNumTrees, int setDepth, T thetaStart, T thetaIncrement, int setNumberOfThetas, double setRandomizationFactor, LPotential *setPotential, T *(LPotential::*setGetTrainValuesSubset)(int, int *, int, int), T (LPotential::*setGetOneEvalValue)(int, int), double setDataRatio) : LLearning(setTrainFolder, setTrainFile, setClassNo)
{
	int i;
	numberOfThetas = setNumberOfThetas;
	numTrees = setNumTrees;
	dataRatio = setDataRatio;

	getTrainValues = NULL;
	getEvalValues = NULL;
	getTrainValuesSubset = setGetTrainValuesSubset;
	getOneEvalValue = setGetOneEvalValue;
	potential = setPotential;

	theta = new T[numberOfThetas];
	theta[0] = thetaStart;
	if(numberOfThetas > 0) theta[1] = thetaStart + thetaIncrement;
	for(i = 2; i < numberOfThetas; i++) theta[i] = theta[i - 1] + theta[i - 2];
	randomizationFactor = setRandomizationFactor, depth = setDepth;
}

template <class T>
LRandomForest<T>::~LRandomForest()
{
	int i;
	if(theta != NULL) delete[] theta;
	for(i = 0; i < randomTrees.GetCount(); i++) delete randomTrees[i];
	randomTrees.Clear();
}


template <class T>
void LRandomForest<T>::SaveTraining()
{
	char *fileName;
	FILE *f;

	fileName = new char[strlen(trainFolder) + strlen(trainFile) + 1];
	sprintf(fileName, "%s%s", trainFolder, trainFile);
	f = fopen(fileName, "wb");

	int weakCount = randomTrees.GetCount();
	fwrite(&weakCount, sizeof(int), 1, f);

	int size = (1 << depth);
	for(int i = 0; i < randomTrees.GetCount(); i++)
	{
		LRandomTree<T> *tree = randomTrees[i];
		fwrite(tree->prob, sizeof(double), classNo * size, f);
		fwrite(tree->theta, sizeof(T), size - 1, f);
		fwrite(tree->index, sizeof(int), size - 1, f);
	}
	fclose(f);
	delete[] fileName;
}

template <class T>
void LRandomForest<T>::LoadTraining()
{
	char *fileName;
	FILE *f;
	int i;

	fileName = new char[strlen(trainFolder) + strlen(trainFile) + 1];
	sprintf(fileName, "%s%s", trainFolder, trainFile);
	f = fopen(fileName, "rb");

	for(i = 0; i < randomTrees.GetCount(); i++) delete randomTrees[i];
	randomTrees.Clear();

	int weakCount;
	fread(&weakCount, sizeof(int), 1, f);

	int size = (1 << depth);
	for(int i = 0; i < weakCount; i++)
	{
		LRandomTree<T> *tree = new LRandomTree<T>(classNo, depth);
		fread(tree->prob, sizeof(double), classNo * size, f);
		fread(tree->theta, sizeof(T), size - 1, f);
		fread(tree->index, sizeof(int), size - 1, f);
		randomTrees.Add(tree);
	}
	fclose(f);
	delete[] fileName;
}

#ifdef MULTITHREAD

template <class T>
struct LRandomForestParams
{
	LRandomForest<T> *forest;
	int *indexes;
	int indexCount;
	int featureFrom, featureTo;
	int *feature;
	T *theta;
	double *entropy;
	unsigned int seed;
	int core;
};

template <class T>
thread_return RandomForestThread(void *par)
{
	LRandomForestParams<T> *params = (LRandomForestParams<T> *)par;
	LMath::SetSeed(params->seed);

	params->forest->FindBestFeature(params->indexes, params->indexCount, params->featureFrom, params->featureTo, params->feature, params->theta, params->entropy, params->core);
	return(thread_defoutput);
}
#endif

template <class T>
void LRandomForest<T>::FindBestFeature(int *indexes, int indexCount, int featureStart, int featureEnd, int *feature, T *thetaBest, double *minEntropy, int core)
{
	int feat, thetaIndex, i;
	LRandomTree<T> optimalSharing(classNo, depth);

	*minEntropy = LMath::positiveInfinity;
	*feature = 0;
	*thetaBest = theta[0];

	double *left = new double[classNo];
	double *right = new double[classNo];

	double *bleft = new double[classNo];
	double *bright = new double[classNo];

	for(feat = featureStart; feat <= featureEnd; feat++)
	{
		if(LMath::RandomReal() < randomizationFactor)
		{
			T *featureValues;
			
			if(getTrainValues != NULL) featureValues = (potential->*getTrainValues)(feat, core);
			else featureValues = (potential->*getTrainValuesSubset)(feat, indexes, indexCount, core);

			for(thetaIndex = 0; thetaIndex < numberOfThetas; thetaIndex++)
			{
				memset(left, 0, classNo * sizeof(double));
				memset(right, 0, classNo * sizeof(double));
				T thetaVal = theta[thetaIndex];

				for(i = 0; i < indexCount; i++)
				{
					if(featureValues[(getTrainValues != NULL) ? (indexes[i]) : i] >= thetaVal) right[targets[indexes[i]]] += weights[indexes[i]][targets[indexes[i]]] * classWeights[targets[indexes[i]]];
					else left[targets[indexes[i]]] += weights[indexes[i]][targets[indexes[i]]] * classWeights[targets[indexes[i]]];
				}
				double sumr = 0, suml = 0;
				for(i = 0; i < classNo; i++) sumr += right[i], suml += left[i];
				
				if((sumr > LMath::almostZero) && (suml > LMath::almostZero)) 
				{
					for(i = 0; i < classNo; i++) right[i] /= sumr;
					for(i = 0; i < classNo; i++) left[i] /= suml;

					double entropy = 0;
					for(i = 0; i < classNo; i++)
					{
						if(left[i] > LMath::almostZero) entropy -= suml * left[i] * log(left[i]);
						if(right[i] > LMath::almostZero) entropy -= sumr * right[i] * log(right[i]);
					}
					if(entropy < *minEntropy)
					{
						*thetaBest = thetaVal;
						*feature = feat;
						*minEntropy = entropy;
						memcpy(bleft, left, classNo * sizeof(double));
						memcpy(bright, right, classNo * sizeof(double));
					}
				}
			}
		}
	}
	if(left != NULL) delete[] left;
	if(right != NULL) delete[] right;
	if(bleft != NULL) delete[] bleft;
	if(bright != NULL) delete[] bright;
}


template <class T>
void LRandomForest<T>::Train(double *setClassWeights, int setNumExamples, int *setTargets, int setNumFeatures, double **setWeights)
{
	int i, j;

	numExamples = setNumExamples, classWeights = setClassWeights, targets = setTargets, numFeatures = setNumFeatures;
	weights = setWeights;

#ifdef MULTITHREAD
	int processors = GetProcessors();
	thread_type *threads;
	LRandomForestParams<T> *params;
	int *minFeatures;
	T *minTheta;
	double *minEntropy;

	threads = new thread_type[processors];	
	memset(threads, 0, processors * sizeof(thread_type));
	params = new LRandomForestParams<T>[processors];
	minFeatures = new int[processors];
	minTheta = new T[processors];
	minEntropy = new double[processors];

	for(i = 0; i < processors; i++)
	{
		params[i].featureFrom = numFeatures * i / processors;
		params[i].featureTo = numFeatures * (i + 1) / processors - 1;
		params[i].feature = &minFeatures[i];
		params[i].theta = &minTheta[i];
		params[i].entropy = &minEntropy[i];
		params[i].forest = this;
		params[i].core = i;
	}
#endif

	printf("Growing trees..\n");
	for(i = 0; i < numTrees; i++)
    {
		printf("Tree no.%d\n", i);

		LRandomTree<T> *tree = new LRandomTree<T>(classNo, depth);

		int **indexesLeft = new int *[depth];
		int **indexesRight = new int *[depth];
		int *indexLeftCount = new int[depth];
		int *indexRightCount = new int[depth];
		memset(indexesLeft, 0, depth * sizeof(int *));
		memset(indexesRight, 0, depth * sizeof(int *));

		indexesLeft[0] = new int[numExamples];

		int dataPos = 0;
		for(j = 0; j < numExamples; j++) if(LMath::RandomReal() <= dataRatio) indexesLeft[0][dataPos] = j, dataPos++;

		indexesRight[0] = NULL;
		indexLeftCount[0] = dataPos;
		indexRightCount[0] = 0;

		int d = 1, pos = 0;
		while(!(pos & 1))
		{
			int realpos = pos >> 1;
			int *indexes = (pos & (1 << (d - 1))) ? indexesRight[d - 1] : indexesLeft[d - 1];
			int indexCount = (pos & (1 << (d - 1))) ? indexRightCount[d - 1] : indexLeftCount[d - 1];

			int featIndex = realpos + (1 << (d - 1)) - 1;

			double minEntropy = LMath::positiveInfinity;
			tree->index[featIndex] = 0, tree->theta[featIndex] = 0;

#ifdef MULTITHREAD
			int running = 0;
			for(j = 0; j < processors; j++)
			{
				params[j].seed = LMath::RandomInt();
				params[j].indexes = indexes;
				params[j].indexCount = indexCount;
				threads[j] = NewThread(RandomForestThread<T>,  &params[j]);
				running += (threads[j] != 0);
			}
		
			while(running)
			{
				for(j = 0; j < processors; j++) if(threads[j])
				{
					if(ThreadFinished(threads[j]))
					{
						CloseThread(&threads[j]);
						running--;

						if(*params[j].entropy < minEntropy)
						{
							minEntropy = *params[j].entropy;
							tree->index[featIndex] = *params[j].feature, tree->theta[featIndex] = *params[j].theta;
						}
					}
				}
				Sleep(0);
			}
#else
			FindBestFeature(indexes, indexCount, 0, numFeatures - 1, &tree->index[featIndex], &tree->theta[featIndex], &minEntropy, 0);
#endif
			T *resp;

			if(getTrainValues != NULL) resp = (potential->*getTrainValues)(tree->index[featIndex], 0);
			else resp = (potential->*getTrainValuesSubset)(tree->index[featIndex], indexes, indexCount, 0);

			if(d < depth)
			{
				indexLeftCount[d] = 0, indexRightCount[d] = 0;
				for(j = 0; j < indexCount; j++)
				{
					if(resp[(getTrainValues != NULL) ? indexes[j] : j] >= tree->theta[featIndex]) indexRightCount[d]++;
					else indexLeftCount[d]++;
				}
				indexesLeft[d] = new int[indexLeftCount[d]];
				indexesRight[d] = new int[indexRightCount[d]];

				if(!indexLeftCount[d]) tree->theta[featIndex] = -(1 << 30);
				else if(!indexRightCount[d]) tree->theta[featIndex] = (1 << 30);

				int lPos = 0, rPos = 0;

				for(j = 0; j < indexCount; j++)
				{
					if(resp[(getTrainValues != NULL) ? indexes[j] : j] >= tree->theta[featIndex]) indexesRight[d][rPos] = indexes[j], rPos++;
					else indexesLeft[d][lPos] = indexes[j], lPos++;
				}
				d++;
			}
			else
			{
				int branchIndex = realpos + (1 << (d - 1));
				for(j = 0; j < indexCount; j++)
				{
					if(resp[(getTrainValues != NULL) ? indexes[j] : j] >= tree->theta[featIndex]) tree->prob[branchIndex * classNo + targets[indexes[j]]] += weights[indexes[j]][targets[indexes[j]]] * classWeights[targets[indexes[j]]];
					else tree->prob[realpos * classNo + targets[indexes[j]]] += weights[indexes[j]][targets[indexes[j]]] * classWeights[targets[indexes[j]]];
				}

				double sum = 0;
				for(j = 0; j < classNo; j++) sum += tree->prob[branchIndex * classNo + j];
				if(sum > LMath::almostZero) for(j = 0; j < classNo; j++) tree->prob[branchIndex * classNo + j] /= sum;

				sum = 0;
				for(j = 0; j < classNo; j++) sum += tree->prob[realpos * classNo + j];
				if(sum > LMath::almostZero) for(j = 0; j < classNo; j++) tree->prob[realpos * classNo + j] /= sum;

				int oldd = d;
				do
				{
					pos -= pos & (1 << d);
					d--;
				}
				while((pos & (1 << d)) && (d > 0));

				pos += (1 << d), d++;

				for(j = oldd - 1; (j >= d); j--)
				{
					if(indexesLeft[j] != NULL)
					{
						delete[] indexesLeft[j];
						indexesLeft[j] = NULL;
					}
					if(indexesRight[j] != NULL)
					{
						delete[] indexesRight[j];
						indexesRight[j] = NULL;
					}
				}
			}
		}
		if(indexesLeft[0] != NULL) delete[] indexesLeft[0];

		if(indexesLeft != NULL) delete[] indexesLeft;
		if(indexesRight != NULL) delete[] indexesRight;
		if(indexLeftCount != NULL) delete[] indexLeftCount;
		if(indexRightCount != NULL) delete[] indexRightCount;

		randomTrees.Add(tree);
	}
#ifdef MULTITHREAD
	if(threads != NULL) delete[] threads;	
	if(params != NULL) delete[] params;
	if(minFeatures != NULL) delete[] minFeatures;
	if(minTheta != NULL) delete[] minTheta;
	if(minEntropy != NULL) delete[] minEntropy;
#endif
}

template <class T>
void LRandomForest<T>::Evaluate(double *costs, int total, int setNumFeatures)
{
	int i, j, k;

	numFeatures = setNumFeatures;
	memset(costs, 0, classNo * total * sizeof(double));

	if(getEvalValues != NULL)
	{
		for(i = 0; i < total; i++) 
		{
			T *resp = (potential->*getEvalValues)(i);
			for(j = 0; j < randomTrees.GetCount(); j++)
			{
				int index = 0;
				LRandomTree<T> *tree = randomTrees[j];
				for(k = 0; k < depth; k++) if(resp[tree->index[index + (1 << k) - 1]] >= tree->theta[index + (1 << k) - 1]) index += (1 << k);
				for(k = 0; k < classNo; k++) costs[i * classNo + k] += tree->prob[index * classNo + k];
			}
			for(k = 0; k < classNo; k++) costs[i * classNo + k] = (costs[i * classNo + k] > LMath::almostZero) ? log(costs[i * classNo + k] / randomTrees.GetCount()) : LMath::negativeInfinity;
		}
	}
	else
	{
		for(i = 0; i < total; i++) 
		{
			for(j = 0; j < randomTrees.GetCount(); j++)
			{
				int index = 0;
				LRandomTree<T> *tree = randomTrees[j];
				for(k = 0; k < depth; k++)
				{
					T resp = (potential->*getOneEvalValue)(tree->index[index + (1 << k) - 1], i);
					if(resp >= tree->theta[index + (1 << k) - 1]) index += (1 << k);
				}
				for(k = 0; k < classNo; k++) costs[i * classNo + k] += tree->prob[index * classNo + k];
			}
			for(k = 0; k < classNo; k++) costs[i * classNo + k] = (costs[i * classNo + k] > LMath::almostZero) ? log(costs[i * classNo + k] / randomTrees.GetCount()) : LMath::negativeInfinity;
		}
	}
}
