#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#endif

#include "std.h"
#include "filter.h"
#include "clustering.h"
#include "feature.h"
#include "crf.h"
#include "learning.h"

template class LList<char>;
template class LList<unsigned char>;
template class LList<int>;
template class LList<unsigned int>;
template class LList<double>;
template class LList<char *>;
template class LList<int *>;
template class LList<double *>;
template class LList<LFilter2D<double> *>;
template class LList<LKMeansClustering<double>::LKdTreeNode *>;
template class LList<LKMeansClustering<double>::LCluster *>;
template class LList<LFeature *>;
template class LList<LDenseFeature *>;
template class LList<LPotential *>;
template class LList<LLearning *>;
template class LList<LCrfLayer *>;
template class LList<LPnCrfLayer *>;
template class LList<LSegmentation2D *>;
template class LList<LBoostWeakLearner<double> *>;
template class LList<LBoostWeakLearner<int> *>;
template class LList<LRandomTree<double> *>;
template class LList<LRandomTree<int> *>;
template class LList<LCrfDomain *>;


void LMath::SetSeed(unsigned int seed)
{
	srand(seed);
}

unsigned int LMath::RandomInt()
{
	return(rand());
}

unsigned int LMath::RandomInt(unsigned int maxval)
{
	return(rand() % maxval);
}

unsigned int LMath::RandomInt(unsigned int minval, unsigned int maxval)
{
	return(minval + RandomInt(maxval - minval));
}

double LMath::RandomReal()
{
	return(rand() / (double)RAND_MAX);
}

double LMath::RandomGaussian(double mi, double var)
{
	double sum = 0;
	int count = 100;
	for(int i = 0; i < count; i++) sum += RandomReal();
	return(mi + sqrt(3 * var / count) * (2 * sum - count));
}

double LMath::SquareEuclidianDistance(double *v1, double *v2, int size)
{
	double dist = 0;
	for(int i = 0; i < size; i++) dist += (v1[i] - v2[i]) * (v1[i] - v2[i]);
	return(dist);
}

double LMath::KLDivergence(double *histogram, double *referenceHistogram, int size, double threshold)
{
	int i;
	double sum1 = (double)0, sum2 = (double)0;
	for(i = 0; i < size; i++) sum1 += histogram[i], sum2 += referenceHistogram[i];
	if(sum1 < almostZero) sum1 = (double)1.0;
	if(sum2 < almostZero) sum2 = (double)1.0;
	
	double value = (double)0;
	for(i = 0; i < size; i++) value += (threshold + ((double)1.0 - threshold) * histogram[i] / sum1) * log((threshold + ((double)1.0 - threshold) * histogram[i] / sum1) / (threshold + ((double)1.0 - threshold) * referenceHistogram[i] / sum2));
	return(value);
}

double LMath::GetAngle(double x, double y)
{
	double r = sqrt(x * x + y * y);
	double angle = (r < almostZero) ? 0 : acos(x / r);
	if(y < 0) angle = 2 * pi - angle;
	return(angle);
}

template <class T> 
LList<T>::LList()
{
	capacity = 1;
	count = 0;

	items = new T[capacity];
}

template <class T>
LList<T>::~LList()
{
	delete[] items;
}

template <class T>
void LList<T>::Resize(int size)
{
	T *newItems;

	newItems = new T[size];
	memcpy(newItems, items, ((capacity < size) ? capacity : size) * sizeof(T));
	delete[] items;
	items = newItems;
	capacity = size;
}

template <class T>
T &LList<T>::operator[](int index)
{
	return(items[index]);
}

template <class T>
T &LList<T>::Add(T value)
{
	if(capacity == count) Resize(capacity << 1);
	items[count] = value;
	count++;

	return(items[count - 1]);
}

template <class T>
T &LList<T>::Insert(T value, int index)
{
	if(capacity == count) Resize(capacity << 1);
	for(int i = count; i > index; i--) items[i] = items[i - 1];
	items[index] = value;
	count++;

	return(items[index]);
}

template <class T>
void LList<T>::Delete(int index)
{
	for(int i = index; i < count - 1; i++) items[i] = items[i + 1];
	count--;
	if((3 * count < capacity) && (capacity > 1)) Resize(capacity >> 1);
}

template <class T>
int LList<T>::GetCount()
{
	return(count);
}

template <class T>
void LList<T>::QuickSort(int from, int to, int (*sort)(T, T))
{
	int i, j;
	T p;

	do
	{
		i = from, j = to, p = items[(from + to) >> 1];
		do
		{
			while(sort(items[i], p) < 0) i++;
			while(sort(items[j], p) > 0) j--;

			if(i <= j)
			{
				if(i != j) Swap(i, j);
				i++, j--;
			}
		}
		while(i <= j);
		if(from < j) QuickSort(from, j, sort);
		from = i;
	}
	while(i <= to);
}

template <class T>
void LList<T>::Sort(int (*sort)(T, T))
{
	if(count > 1) QuickSort(0, count - 1, sort);
}

template <class T>
void LList<T>::Swap(int index1, int index2)
{
	T tmp = items[index1];
	items[index1] = items[index2];
	items[index2] = tmp;
}

template <class T>
T *LList<T>::GetArray()
{
	return(items);
}

template <class T>
void LList<T>::Clear()
{
	if(capacity > 1) Resize(1);
	count = 0;
}

void _error(char *str)
{
	printf("%s\n", str);
	exit(1);
}

char *GetFileName(const char *folder, const char *name, const char *extension)
{
	char *fileName;
	fileName = new char[strlen(folder) + strlen(name) + strlen(extension) + 1];
	sprintf(fileName, "%s%s%s", folder, name, extension);
	return(fileName);
}

void ForceDirectory(const char *dir)
{
	char *dirNew = new char[strlen(dir) + 1];
	strcpy(dirNew, dir);
	int i = 0;
	while(i != strlen(dirNew))
	{
		i++;
		if((dirNew[i] == '/') || (dirNew[i] == '\\'))
		{
			char old = dirNew[i + 1];
			dirNew[i + 1] = 0;
#ifdef _WIN32			
			if(GetFileAttributes(dirNew) == INVALID_FILE_ATTRIBUTES) CreateDirectory(dirNew, NULL);
#else
			mkdir(dirNew, S_IRWXU | S_IRWXG);
#endif			
			dirNew[i + 1] = old;
		}
	}
	if(dirNew != NULL) delete[] dirNew;
}

void ForceDirectory(const char *dir, const char *subdir)
{
	char *path = new char[strlen(dir) + strlen(subdir) + 1];
	sprintf(path, "%s%s", dir, subdir);
	ForceDirectory(path);
	delete[] path;
}

int GetProcessors()
{
#ifdef _WIN32			
	SYSTEM_INFO sysInfo;
	GetSystemInfo(&sysInfo);
	int processors = sysInfo.dwNumberOfProcessors;
#else
	int processors = sysconf(_SC_NPROCESSORS_CONF);
#endif
	if(processors > MAXTHREAD) processors = MAXTHREAD;
	return(processors);
}

#ifdef MULTITHREAD
#ifdef _WIN32
CRITICAL_SECTION critSection;

void InitializeCriticalSection()
{
	InitializeCriticalSection(&critSection);
}
void DeleteCriticalSection()
{
	DeleteCriticalSection(&critSection);
}
void EnterCriticalSection()
{
	EnterCriticalSection(&critSection);
}
void LeaveCriticalSection()
{
	LeaveCriticalSection(&critSection);
}

thread_type NewThread(thread_return (*routine)(void *), void *param)
{
	DWORD id;
	return(CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)routine,  param, 0, &id));
}

int ThreadFinished(thread_type thread)
{
	return(WaitForSingleObject(thread, 0) == WAIT_OBJECT_0);
}

void CloseThread(thread_type *thread)
{
    CloseHandle(*thread);
	*thread = 0;
}

#else
pthread_mutex_t mutex;

void InitializeCriticalSection()
{
	pthread_mutex_init(&mutex, NULL);
}
void DeleteCriticalSection()
{
	pthread_mutex_destroy(&mutex);
}
void EnterCriticalSection()
{
	pthread_mutex_lock(&mutex);
}
void LeaveCriticalSection()
{
	pthread_mutex_unlock(&mutex);
}

void Sleep(int time)
{
	sleep(time);
}

thread_type NewThread(thread_return (*routine)(void *), void *param)
{
	thread_type thread;
	if(pthread_create(&thread, NULL, routine, param)) return(0);
	else return(thread);
}

int ThreadFinished(thread_type thread)
{
	return(!pthread_tryjoin_np(thread, NULL));
}

void CloseThread(thread_type *thread)
{
	*thread = 0;
}
#endif
#endif

