#ifndef __filter
#define __filter

#include "std.h"
#include "image.h"

template <class T> class LImage;
template <class T> class LColourImage;

template <class T>
class LMaskFilter1D
{
	protected :
		int centre, size;
		double *data;
	public :
		LMaskFilter1D(int setCentre);
		virtual ~LMaskFilter1D();

		void ConvolveX(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int sampleSize, int extend);
		void ConvolveY(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int sampleSize, int extend);
		void ConvolvePartX(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int extend, int x0, int y0);
		void ConvolvePartY(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int extend, int x0, int y0);
		T Response(T *values);
		void Response(T *values, int bands, T *out);
		int GetCentre();
};

template <class T>
class LGaussianFilter1D : public LMaskFilter1D<T>
{
	public :
		LGaussianFilter1D(double sigma);
};

template <class T>
class LGaussianDerivativeFilter1D : public LMaskFilter1D<T>
{
	public :
		LGaussianDerivativeFilter1D(double sigma);
};

template <class T>
class LGaussian2ndDerivativeFilter1D : public LMaskFilter1D<T>
{
	public :
		LGaussian2ndDerivativeFilter1D(double sigma);
};

template <class T>
class LFilter2D
{
	protected :
		int bands;
	public :
		LFilter2D(int setBands);
		virtual ~LFilter2D() {};

		int GetBands();
		virtual void Filter(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int sampleSize) = 0;
};

template <class T>
class LMaskFilter2D : public LFilter2D<T>
{
	protected :
		virtual void Convolve(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int sampleSize, int extend) = 0;
		virtual void ConvolvePart(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int extend, int x0, int y0) = 0;
	public :
		LMaskFilter2D(int setBands);

		void Filter(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int sampleSize);
		void FilterPart(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int x0, int y0);
		virtual void Response(LColourImage<T> &imageFrom, int x, int y, T *out) = 0;
};

template <class T>
class LSeparableFilter2D : public LMaskFilter2D<T>
{
	private :
		LMaskFilter1D<T> *filterX, *filterY;
	protected :
		void Set1DFilters(LMaskFilter1D<T> *setFilterX, LMaskFilter1D<T> *setFilterY);
		void Convolve(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int sampleSize, int extend); 
		void ConvolvePart(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int extend, int x0, int y0); 
	public :
		LSeparableFilter2D(int setBands);
		LSeparableFilter2D(LMaskFilter1D<T> *setFilterX, LMaskFilter1D<T> *setFilterY, int setBands);
		~LSeparableFilter2D();
		void Response(LColourImage<T> &imageFrom, int x, int y, T *out);
		void Response(LColourImage<T> &imageFrom, double x, double y, T *out);
};

template <class T>
class LLogFilter2D : public LMaskFilter2D<T>
{
	private :
		LMaskFilter1D<T> *filter1, *filter2;
	protected :
		void Set1DFilters(LMaskFilter1D<T> *setFilter1, LMaskFilter1D<T> *setFilter2);
		void Convolve(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int sampleSize, int extend); 
		void ConvolvePart(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int extend, int x0, int y0); 
	public :
		LLogFilter2D(int setBands);
		LLogFilter2D(LMaskFilter1D<T> *setFilter1, LMaskFilter1D<T> *setFilter2, int setBands);
		~LLogFilter2D();
		void Response(LColourImage<T> &imageFrom, int x, int y, T *out);
		void Response(LColourImage<T> &imageFrom, double x, double y, T *out);
};

template <class T>
class LGaussianFilter2D : public LSeparableFilter2D<T>
{
	public :
		LGaussianFilter2D(double sigma, int setBands);
};

template <class T>
class LGaussianDerivativeXFilter2D : public LSeparableFilter2D<T>
{
	public :
		LGaussianDerivativeXFilter2D(double sigmaX, double sigmaY, int setBands);
};

template <class T>
class LGaussianDerivativeYFilter2D : public LSeparableFilter2D<T>
{
	public :
		LGaussianDerivativeYFilter2D(double sigmaX, double sigmaY, int setBands);
};

template <class T>
class LGaussianDerivativeXYFilter2D : public LSeparableFilter2D<T>
{
	public :
		LGaussianDerivativeXYFilter2D(double sigmaX, double sigmaY, int setBands);
};

template <class T>
class LLaplacianFilter2D : public LLogFilter2D<T>
{
	public :
		LLaplacianFilter2D(double sigma, int setBands);
};

template <class T>
class LMeanShiftFilter2D : public LFilter2D<T>
{
	private :
		int skipNeighbours;
		double sigmaXY, sigmaLuv;
		void CalculateMeanDiff(LImage<T> &image, double *point, double *meanDiff, double *meanAbs, double *rangeMins, int *buckets, int *noBuckets, int *bucketNeighs, int *bucketList);
	public :
		LMeanShiftFilter2D(double setSigmaXY, double setSigmaLuv, int setSkipNeighbours, int setBands);

		void SetParameters(double setSigmaXY, double setSigmaLuv, int setSkipNeighbours);
		void Filter(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int sampleSize);
};

#endif