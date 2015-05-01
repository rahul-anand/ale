#include <stdio.h>
#include <math.h>
#include <string.h>
#include "filter.h"

template class LMaskFilter1D<double>;
template class LGaussianFilter1D<double>;
template class LGaussianDerivativeFilter1D<double>;
template class LGaussian2ndDerivativeFilter1D<double>;
template class LFilter2D<double>;
template class LMaskFilter2D<double>;
template class LSeparableFilter2D<double>;
template class LLogFilter2D<double>;
template class LGaussianFilter2D<double>;
template class LGaussianDerivativeXFilter2D<double>;
template class LGaussianDerivativeYFilter2D<double>;
template class LGaussianDerivativeXYFilter2D<double>;
template class LLaplacianFilter2D<double>;
template class LMeanShiftFilter2D<double>;

template class LMaskFilter1D<unsigned char>;
template class LGaussianFilter1D<unsigned char>;
template class LGaussianDerivativeFilter1D<unsigned char>;
template class LGaussian2ndDerivativeFilter1D<unsigned char>;
template class LFilter2D<unsigned char>;
template class LMaskFilter2D<unsigned char>;
template class LSeparableFilter2D<unsigned char>;
template class LLogFilter2D<unsigned char>;
template class LGaussianFilter2D<unsigned char>;
template class LGaussianDerivativeXFilter2D<unsigned char>;
template class LGaussianDerivativeYFilter2D<unsigned char>;
template class LGaussianDerivativeXYFilter2D<unsigned char>;
template class LLaplacianFilter2D<unsigned char>;

template <class T>
LMaskFilter1D<T>::LMaskFilter1D(int setCentre)
{
	centre = setCentre, size = 2 * setCentre + 1;
	data = new double[size];
}

template <class T>
LMaskFilter1D<T>::~LMaskFilter1D()
{
	if(data != NULL) delete[] data;
}

template <class T>
void LMaskFilter1D<T>::ConvolveX(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int sampleSize, int extend)
{
	int i, j, k, width, height, bandsFrom, bandsTo, min, max, filterPos;
	T *dataFrom, *dataTo;

	width = imageFrom.GetWidth();
	height = imageFrom.GetHeight();
	bandsFrom = imageFrom.GetBands();
	bandsTo = imageTo.GetBands();
	dataTo = &imageTo(0, 0, bandTo);

	for(i = 0; i < height; i++) for(j = 0; j < width; j += sampleSize, dataTo += bandsTo)
	{
		min = (j - centre < 0) ? 0 : j - centre;
		max = (j + centre > width - 1) ? width - 1 : j + centre;

		dataFrom = &imageFrom(min, i, bandFrom);

		double sum = 0;
		filterPos = 0;

		if(j - centre < 0)
		{
			if(extend) for(k = j - centre; k < 0; k++, filterPos++) sum += data[filterPos] * (*dataFrom);
			else filterPos = centre - j;
		}
		for(k = min; k <= max; k++, filterPos++, dataFrom += bandsFrom) sum += data[filterPos] * (*dataFrom);

		if((j + centre > width - 1) && (extend))
		{
			dataFrom -= bandsFrom;
			for(k = width; k < j + centre; k++, filterPos++) sum += data[filterPos] * (*dataFrom);
		}

		*dataTo = (T)sum;
	}
}

template <class T>
void LMaskFilter1D<T>::ConvolveY(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int sampleSize, int extend)
{
	int i, j, k, width, height, bandsFrom, bandsTo, min, max, filterPos;
	T *dataFrom, *dataTo;

	width = imageFrom.GetWidth();
	height = imageFrom.GetHeight();
	bandsFrom = imageFrom.GetBands();
	bandsTo = imageTo.GetBands();
	dataTo = &imageTo(0, 0, bandTo);

	for(i = 0; i < height; i+= sampleSize) for(j = 0; j < width; j++, dataTo += bandsTo)
	{
		min = (i - centre < 0) ? 0 : i - centre;
		max = (i + centre > height - 1) ? height - 1 : i + centre;

		dataFrom = &imageFrom(j, min, bandFrom);

		double sum = 0;
		filterPos = 0;

		if(i - centre < 0)
		{
			if(extend) for(k = i - centre; k < 0; k++, filterPos++) sum += data[filterPos] * (*dataFrom);
			else filterPos = centre - i;
		}
		for(k = min; k <= max; k++, filterPos++, dataFrom += bandsFrom * width) sum += data[filterPos] * (*dataFrom);
		if((i + centre > height - 1) && (extend))
		{
			dataFrom -= bandsFrom * width;
			for(k = height; k < i + centre; k++, filterPos++) sum += data[filterPos] * (*dataFrom);
		}
		*dataTo = (T)sum;
	}
}

template <class T>
void LMaskFilter1D<T>::ConvolvePartX(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int extend, int x0, int y0)
{
	int i, j, k, width, height, bandsFrom, bandsTo, min, max, filterPos, widthTo, heightTo;
	T *dataFrom, *dataTo;

	width = imageFrom.GetWidth();
	height = imageFrom.GetHeight();
	bandsFrom = imageFrom.GetBands();
	bandsTo = imageTo.GetBands();
	dataTo = &imageTo(0, 0, bandTo);
	widthTo = imageTo.GetWidth();
	heightTo = imageTo.GetHeight();

	for(i = y0; i < y0 + heightTo; i++) for(j = x0; j < x0 + widthTo; j++, dataTo += bandsTo)
	{
		min = (j - centre < 0) ? 0 : j - centre;
		max = (j + centre > width - 1) ? width - 1 : j + centre;

		dataFrom = &imageFrom(min, i, bandFrom);

		double sum = 0;
		filterPos = 0;

		if(j - centre < 0)
		{
			if(extend) for(k = j - centre; k < 0; k++, filterPos++) sum += data[filterPos] * (*dataFrom);
			else filterPos = centre - j;
		}

		for(k = min; k <= max; k++, filterPos++, dataFrom += bandsFrom) sum += data[filterPos] * (*dataFrom);

		if(j - centre < width)
		{
			if((j + centre > width - 1) && (extend))
			{
				dataFrom -= bandsFrom;
				for(k = width; k < j + centre; k++, filterPos++) sum += data[filterPos] * (*dataFrom);
			}
		}
		else sum = 0;

		*dataTo = (T)sum;
	}
}

template <class T>
void LMaskFilter1D<T>::ConvolvePartY(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int extend, int x0, int y0)
{
	int i, j, k, width, height, bandsFrom, bandsTo, min, max, filterPos, widthTo, heightTo;
	T *dataFrom, *dataTo;

	width = imageFrom.GetWidth();
	height = imageFrom.GetHeight();
	bandsFrom = imageFrom.GetBands();
	bandsTo = imageTo.GetBands();
	dataTo = &imageTo(0, 0, bandTo);
	widthTo = imageTo.GetWidth();
	heightTo = imageTo.GetHeight();

	for(i = y0; (i < y0 + heightTo) && (i < height); i++) for(j = x0; j < x0 + widthTo; j++, dataTo += bandsTo)
	{
		min = (i - centre < 0) ? 0 : i - centre;
		max = (i + centre > height - 1) ? height - 1 : i + centre;

		dataFrom = &imageFrom(j, min, bandFrom);

		double sum = 0;
		filterPos = 0;

		if(i - centre < 0)
		{
			if(extend) for(k = i - centre; k < 0; k++, filterPos++) sum += data[filterPos] * (*dataFrom);
			else filterPos = centre - i;
		}
		for(k = min; k <= max; k++, filterPos++, dataFrom += bandsFrom * width) sum += data[filterPos] * (*dataFrom);

		if(i - centre < height)
		{
			if((i + centre > height - 1) && (extend))
			{
				dataFrom -= bandsFrom * width;
				for(k = height; k < i + centre; k++, filterPos++) sum += data[filterPos] * (*dataFrom);
			}
		}
		else sum = 0;
		*dataTo = (T)sum;
	}
}

template <class T>
T LMaskFilter1D<T>::Response(T *values)
{
	double sum = 0;
	for(int i = 0; i < size; i++) sum += data[i] * values[i];
	return((T)sum);
}

template <class T>
void LMaskFilter1D<T>::Response(T *values, int bands, T *out)
{
	for(int j = 0; j < bands; j++)
	{
		double sum = 0;
		for(int i = 0; i < size; i++) sum += data[i] * values[bands * i + j];
		out[j] = (T)sum;
	}
}

template <class T>
int LMaskFilter1D<T>::GetCentre()
{
	return(centre);
}

template <class T>
LGaussianFilter1D<T>::LGaussianFilter1D(double sigma) : LMaskFilter1D<T>((int)(3 * sigma))
{
	double c1  = (double)1.0 / ((double)sqrt((double)2.0 * LMath::pi) * sigma), c2 = 1 / ((double)2.0 * sigma * sigma), sum = 0;
	int i;

	for(i = 0; i < this->size; i++)
    {
		int x = i - this->centre;
		this->data[i] = c1 * (double)exp(-x * x * c2);
		sum += this->data[i];
    }
	for(i = 0; i < this->size; i++) this->data[i] /= sum;
}

template <class T>
LGaussianDerivativeFilter1D<T>::LGaussianDerivativeFilter1D(double sigma) : LMaskFilter1D<T>((int)(3 * sigma))
{
	double c1 = (double)1.0 / ((double)sqrt((double)2.0 * LMath::pi) * sigma * sigma * sigma);
	double c2 = (double)1.0 / ((double)2.0 * sigma * sigma);

	for(int i = 0; i < this->size; i++)
    {
		int x = i - this->centre;
		this->data[i] = -c1 * x * (double)exp(-x * x * c2);
    }
}

template <class T>
LGaussian2ndDerivativeFilter1D<T>::LGaussian2ndDerivativeFilter1D(double sigma) : LMaskFilter1D<T>((int)(3 * sigma))
{
	double c1 = (double)1.0 / ((double)sqrt((double)2.0 * LMath::pi) * sigma);
	double c2 = (double)1.0 / ((double)2.0 * sigma * sigma);
	double c3 = (double)1.0 / ((double)sigma * sigma), sum = 0;
	int i;

	for(i = 0; i < this->size; i++)
    {
		int x = i - this->centre;
		this->data[i] = c1 * ((double)x * x * c3 * c3 - c3) * exp(-x * x * c2);
		sum += this->data[i];
    }
	sum /= this->size;
	for(i = 0; i < this->size; i++) this->data[i] -= sum;
}

template <class T>
LFilter2D<T>::LFilter2D(int setBands)
{
	this->bands = setBands;
}

template <class T>
int LFilter2D<T>::GetBands()
{
	return(this->bands);
}

template <class T>
LMaskFilter2D<T>::LMaskFilter2D(int setBands) : LFilter2D<T>(setBands)
{
}

template <class T>
void LMaskFilter2D<T>::Filter(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int sampleSize)
{
	for(int i = 0; i < this->bands; i++) Convolve(imageFrom, bandFrom + i, imageTo, bandTo + i, sampleSize, 1);
}

template <class T>
void LMaskFilter2D<T>::FilterPart(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int x0, int y0)
{
	for(int i = 0; i < this->bands; i++) ConvolvePart(imageFrom, bandFrom + i, imageTo, bandTo + i, 1, x0, y0);
}


template <class T>
LSeparableFilter2D<T>::LSeparableFilter2D(int setBands) : LMaskFilter2D<T>(setBands)
{
	filterX = NULL, filterY = NULL;
}

template <class T>
LSeparableFilter2D<T>::LSeparableFilter2D(LMaskFilter1D<T> *setFilterX, LMaskFilter1D<T> *setFilterY, int setBands) : LMaskFilter2D<T>(setBands)
{
	filterX = setFilterX, filterY = setFilterY;
}

template <class T>
void LSeparableFilter2D<T>::Set1DFilters(LMaskFilter1D<T> *setFilterX, LMaskFilter1D<T> *setFilterY)
{
	if(filterX != NULL) delete(filterX);
	if(filterY != NULL) delete(filterY);

	filterX = setFilterX, filterY = setFilterY;
}

template <class T>
LSeparableFilter2D<T>::~LSeparableFilter2D()
{
	if(filterX != NULL) delete(filterX);
	if(filterY != NULL) delete(filterY);
}

template <class T>
void LSeparableFilter2D<T>::Convolve(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int sampleSize, int extend)
{
	LImage<T> temp((imageFrom.GetWidth() + sampleSize - 1) / sampleSize, imageFrom.GetHeight(), 1);
	filterX->ConvolveX(imageFrom, bandFrom, temp, 0, sampleSize, extend);
	filterY->ConvolveY(temp, 0, imageTo, bandTo, sampleSize, extend);
}

template <class T>
void LSeparableFilter2D<T>::ConvolvePart(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int extend, int x0, int y0)
{
	LImage<T> temp(imageTo.GetWidth(), imageFrom.GetHeight(), 1);
	filterX->ConvolvePartX(imageFrom, bandFrom, temp, 0, extend, x0, 0);
	filterY->ConvolvePartY(temp, 0, imageTo, bandTo, extend, 0, y0);
}

template <class T>
void LSeparableFilter2D<T>::Response(LColourImage<T> &imageFrom, int x, int y, T *out)
{
	T *input, *output;
	int xCentre = filterX->GetCentre(), yCentre = filterY->GetCentre();
	int xSize = 2 * xCentre + 1, ySize = yCentre * 2 + 1, i, j;
	int bands = imageFrom.GetBands();

	input = new T[xSize * bands];
	output = new T[ySize * bands];

	int height = imageFrom.GetHeight();
	int width = imageFrom.GetWidth();

	for(i = -yCentre; i <= yCentre; i++)
	{
		int y0 = (y + i < 0) ? 0 : ((y + i > height - 1) ? height - 1 : y + i);
		int xFrom = x - xCentre, xTo = x + xCentre + 1;
		xFrom = (xFrom < 0) ? 0 : ((xFrom > width - 1) ? width - 1 : xFrom);
		xTo = (xTo < 0) ? 0 : ((xTo > width - 1) ? width - 1 : xTo);

		for(j = 0; (j < xFrom - x + xCentre) && (j < xSize); j++) memcpy(input + j * bands, imageFrom(xFrom, y0), bands * sizeof(T));
		for(j = xSize - 1; (j > x + xCentre + xSize - 1 - xTo) && (j >= 0); j--) memcpy(input + j * bands, imageFrom(xTo, y0), bands * sizeof(T));
		if(xTo > xFrom) memcpy(input + (xFrom - x + xCentre) * bands, imageFrom(xFrom, y0), bands * (xTo - xFrom) * sizeof(T));
		filterX->Response(input, bands, output + (i + yCentre) * bands);
	}
	filterY->Response(output, bands, out);
	delete[] input;
	delete[] output;
}

template <class T>
void LSeparableFilter2D<T>::Response(LColourImage<T> &imageFrom, double x, double y, T *out)
{
	T *input, *output;
	int xCentre = filterX->GetCentre(), yCentre = filterY->GetCentre();
	int xSize = 2 * xCentre + 1, ySize = yCentre * 2 + 1, i, j, k;
	int bands = imageFrom.GetBands();

	input = new T[xSize * bands];
	output = new T[ySize * bands];

	int height = imageFrom.GetHeight();
	int width = imageFrom.GetWidth();

	for(i = -yCentre; i <= yCentre; i++)
	{
		double y0 = (y + i < 0) ? 0 : ((y + i > height - 1) ? height - 1 : y + i);
		int y1 = (int)y0, y2 = (y1 == height - 1) ? y1 : y1 + 1;
		double coefY = 1 - y0 + y1;

		for(j = -xCentre; j <= xCentre; j++)
		{
			double x0 = (x + j < 0) ? 0 : ((x + j > width - 1) ? width - 1 : x + j);
			int x1 = (int)x0, x2 = (x1 == width - 1) ? x1 : x1 + 1;
			double coefX = 1 - x0 + x1;
			for(k = 0; k < bands; k++) input[(j + xCentre) * bands + k] = (T)(coefY * (coefX * imageFrom(x1, y1, k) + (1 - coefX) * imageFrom(x2, y1, k)) + (1 - coefY) * (coefX * imageFrom(x1, y2, k) + (1 - coefX) * imageFrom(x2, y2, k)));
		}
		filterX->Response(input, bands, output + (i + yCentre) * bands);
	}
	filterY->Response(output, bands, out);
	delete[] input;
	delete[] output;
}

template <class T>
LLogFilter2D<T>::LLogFilter2D(int setBands) : LMaskFilter2D<T>(setBands)
{
	filter1 = NULL, filter2 = NULL;
}

template <class T>
LLogFilter2D<T>::LLogFilter2D(LMaskFilter1D<T> *setFilter1, LMaskFilter1D<T> *setFilter2, int setBands) : LMaskFilter2D<T>(setBands)
{
	filter1 = setFilter1, filter2 = setFilter2;
}

template <class T>
void LLogFilter2D<T>::Set1DFilters(LMaskFilter1D<T> *setFilter1, LMaskFilter1D<T> *setFilter2)
{
	if(filter1 != NULL) delete(filter1);
	if(filter2 != NULL) delete(filter2);

	filter1 = setFilter1, filter2 = setFilter2;
}

template <class T>
LLogFilter2D<T>::~LLogFilter2D()
{
	if(filter1 != NULL) delete(filter1);
	if(filter2 != NULL) delete(filter2);
}

template <class T>
void LLogFilter2D<T>::Convolve(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int sampleSize, int extend)
{
	LImage<T> temp1((imageFrom.GetWidth() + sampleSize - 1) / sampleSize, imageFrom.GetHeight(), 1);
	LImage<T> temp2((imageFrom.GetWidth() + sampleSize - 1) / sampleSize, (imageFrom.GetHeight() + sampleSize - 1) / sampleSize, 1);
	LImage<T> temp3(imageFrom.GetWidth(), (imageFrom.GetHeight() + sampleSize - 1) / sampleSize, 1);
	T *dataFrom, *dataTo;
	int size, bands;

	filter1->ConvolveX(imageFrom, bandFrom, temp1, 0, sampleSize, extend);
	filter2->ConvolveY(temp1, 0, temp2, 0, sampleSize, extend);

	filter1->ConvolveY(imageFrom, bandFrom, temp3, 0, sampleSize, extend);
	filter2->ConvolveX(temp3, 0, imageTo, bandTo, sampleSize, extend);

	dataFrom = temp2.GetData();
	dataTo = &imageTo(0, 0, bandTo);
	size = imageTo.GetWidth() * imageTo.GetHeight();
	bands = imageTo.GetBands();

	for(int i = 0; i < size; i++, dataFrom++, dataTo += bands) *dataTo += *dataFrom;
}

template <class T>
void LLogFilter2D<T>::ConvolvePart(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int extend, int x0, int y0)
{
	LImage<T> temp1(imageTo.GetWidth(), imageFrom.GetHeight(), 1);
	LImage<T> temp2(imageTo.GetWidth(), imageTo.GetHeight(), 1);
	LImage<T> temp3(imageFrom.GetWidth(), imageTo.GetHeight(), 1);
	T *dataFrom, *dataTo;
	int size, bands;

	filter1->ConvolvePartX(imageFrom, bandFrom, temp1, 0, extend, x0, 0);
	filter2->ConvolvePartY(temp1, 0, temp2, 0, extend, 0, y0);

	filter1->ConvolvePartY(imageFrom, bandFrom, temp3, 0, extend, 0, y0);
	filter2->ConvolvePartX(temp3, 0, imageTo, bandTo, extend, x0, 0);

	dataFrom = temp2.GetData();
	dataTo = &imageTo(0, 0, bandTo);
	size = imageTo.GetWidth() * imageTo.GetHeight();
	bands = imageTo.GetBands();

	for(int i = 0; i < size; i++, dataFrom++, dataTo += bands) *dataTo += *dataFrom;
}

template <class T>
void LLogFilter2D<T>::Response(LColourImage<T> &imageFrom, int x, int y, T *out)
{
	T *input, *output, *output2, *out2;
	int centre = filter1->GetCentre();
	int size = 2 * centre + 1, i, j;
	int bands = imageFrom.GetBands();

	out2 = new T[bands];
	input = new T[size * bands];
	output = new T[size * bands];
	output2 = new T[size * bands];

	int height = imageFrom.GetHeight();
	int width = imageFrom.GetWidth();

	for(i = -centre; i <= centre; i++)
	{
		int y0 = (y + i < 0) ? 0 : ((y + i > height - 1) ? height - 1 : y + i);
		int xFrom = x - centre, xTo = x + centre + 1;
		xFrom = (xFrom < 0) ? 0 : ((xFrom > width - 1) ? width - 1 : xFrom);
		xTo = (xTo < 0) ? 0 : ((xTo > width - 1) ? width - 1 : xTo);

		for(j = 0; (j < xFrom - x + centre) && (j < size); j++) memcpy(input + j * bands, imageFrom(xFrom, y0), bands * sizeof(T));
		for(j = size - 1; (j > x + centre + size - 1 - xTo) && (j >= 0); j--) memcpy(input + j * bands, imageFrom(xTo, y0), bands * sizeof(T));
		if(xTo > xFrom) memcpy(input + (xFrom - x + centre) * bands, imageFrom(xFrom, y0), bands * (xTo - xFrom) * sizeof(T));
		filter1->Response(input, bands, output + (i + centre) * bands);
		filter2->Response(input, bands, output2 + (i + centre) * bands);
	}
	filter2->Response(output, bands, out);
	filter1->Response(output2, bands, out2);
	for(i = 0; i < bands; i++) out[i] += out2[i];

	delete[] input;
	delete[] output;
	delete[] output2;
	delete[] out2;
}

template <class T>
void LLogFilter2D<T>::Response(LColourImage<T> &imageFrom, double x, double y, T *out)
{
	T *input, *output, *output2, *out2;
	int centre = filter1->GetCentre();
	int size = 2 * centre + 1, i, j, k;
	int bands = imageFrom.GetBands();

	out2 = new T[bands];
	input = new T[size * bands];
	output = new T[size * bands];
	output2 = new T[size * bands];

	int height = imageFrom.GetHeight();
	int width = imageFrom.GetWidth();

	for(i = -centre; i <= centre; i++)
	{
		double y0 = (y + i < 0) ? 0 : ((y + i > height - 1) ? height - 1 : y + i);
		int y1 = (int)y0, y2 = (y1 == height - 1) ? y1 : y1 + 1;
		double coefY = 1 - y0 + y1;

		for(j = -centre; j <= centre; j++)
		{
			double x0 = (x + j < 0) ? 0 : ((x + j > width - 1) ? width - 1 : x + j);
			int x1 = (int)x0, x2 = (x1 == width - 1) ? x1 : x1 + 1;
			double coefX = 1 - x0 + x1;
			for(k = 0; k < bands; k++) input[(j + centre) * bands + k] = (T)(coefY * (coefX * imageFrom(x1, y1, k) + (1 - coefX) * imageFrom(x2, y1, k)) + (1 - coefY) * (coefX * imageFrom(x1, y2, k) + (1 - coefX) * imageFrom(x2, y2, k)));
		}
		filter1->Response(input, bands, output + (i + centre) * bands);
		filter2->Response(input, bands, output2 + (i + centre) * bands);
	}
	filter2->Response(output, bands, out);
	filter1->Response(output2, bands, out2);
	for(i = 0; i < bands; i++) out[i] += out2[i];

	delete[] input;
	delete[] output;
	delete[] output2;
	delete[] out2;
}

template <class T>
LGaussianFilter2D<T>::LGaussianFilter2D(double sigma, int setBands) : LSeparableFilter2D<T>(setBands)
{
	this->Set1DFilters(new LGaussianFilter1D<T>(sigma), new LGaussianFilter1D<T>(sigma));
}

template <class T>
LGaussianDerivativeXFilter2D<T>::LGaussianDerivativeXFilter2D(double sigmaX, double sigmaY, int setBands) : LSeparableFilter2D<T>(setBands)
{
	this->Set1DFilters(new LGaussianDerivativeFilter1D<T>(sigmaX), new LGaussianFilter1D<T>(sigmaY));
}

template <class T>
LGaussianDerivativeYFilter2D<T>::LGaussianDerivativeYFilter2D(double sigmaX, double sigmaY, int setBands) : LSeparableFilter2D<T>(setBands)
{
	this->Set1DFilters(new LGaussianFilter1D<T>(sigmaX), new LGaussianDerivativeFilter1D<T>(sigmaY));
}

template <class T>
LGaussianDerivativeXYFilter2D<T>::LGaussianDerivativeXYFilter2D(double sigmaX, double sigmaY, int setBands) : LSeparableFilter2D<T>(setBands)
{
	this->Set1DFilters(new LGaussianDerivativeFilter1D<T>(sigmaX), new LGaussianDerivativeFilter1D<T>(sigmaY));
}

template <class T>
LLaplacianFilter2D<T>::LLaplacianFilter2D(double sigma, int setBands) : LLogFilter2D<T>(setBands)
{
	this->Set1DFilters(new LGaussian2ndDerivativeFilter1D<T>(sigma), new LGaussianFilter1D<T>(sigma));
}

template <class T>
LMeanShiftFilter2D<T>::LMeanShiftFilter2D(double setSigmaXY, double setSigmaLuv, int setSkipNeighbours, int setBands) : LFilter2D<T>(setBands)
{
	sigmaXY = setSigmaXY;
	sigmaLuv = setSigmaLuv;
	skipNeighbours = setSkipNeighbours;
}

template <class T>
void LMeanShiftFilter2D<T>::SetParameters(double setSigmaXY, double setSigmaLuv, int setSkipNeighbours)
{
	sigmaXY = setSigmaXY;
	sigmaLuv = setSigmaLuv;
	skipNeighbours = setSkipNeighbours;
}

template <class T>
void LMeanShiftFilter2D<T>::CalculateMeanDiff(LImage<T> &image, double *point, double *meanDiff, double *meanAbs, double *rangeMins, int *buckets, int *noBuckets, int *bucketNeighs, int *bucketList)
{
	int j, k, pointCount = 0;
	int bands = image.GetBands();

	for (j = 0; j < bands; j++) meanDiff[j] = 0;
	int neighSum = 0;
	int bucketIndex = (int)(point[0] - rangeMins[0]) + 1 + noBuckets[0] * ((int)(point[1] - rangeMins[1]) + 1 + noBuckets[1] * ((int)(point[2] - rangeMins[2]) + 1));

	for(j = 0; j < 27; j++)
	{
		int neighIndex = buckets[bucketIndex + bucketNeighs[j]];
		while(neighIndex >= 0)
		{
			T *neighData = image.GetData() + bands * neighIndex;
			double diff = (neighData[0] - point[0]) * (neighData[0] - point[0]) + (neighData[1] - point[1]) * (neighData[1] - point[1]);

			if(diff < 1.0)
			{
				diff = (neighData[2] - point[2]) * (neighData[2] - point[2]);
				if(point[2] > 80.0 / sigmaLuv) diff *= 4;
				for(k = 3; k < bands; k++) diff += (neighData[k] - point[k]) * (neighData[k] - point[k]);

				if(diff < 1.0)
				{
					for(k = 0; k < bands; k++) meanDiff[k] += neighData[k];
					neighSum++;
				}
			}
	        neighIndex = bucketList[neighIndex];
		}
	}
	if(neighSum > 0) for(j = 0; j < bands; j++) meanDiff[j] = meanDiff[j] / neighSum - point[j];
	else for(j = 0; j < bands; j++) meanDiff[j] = 0;

	*meanAbs = (meanDiff[0] * meanDiff[0] + meanDiff[1] * meanDiff[1]) * sigmaXY * sigmaXY;
	for(j = 2; j < bands; j++) *meanAbs += meanDiff[j] * meanDiff[j] * sigmaLuv * sigmaLuv;
}

template <class T>
void LMeanShiftFilter2D<T>::Filter(LImage<T> &imageFrom, int bandFrom, LImage<T> &imageTo, int bandTo, int sampleSize)
{
	int width, height, points, i, j, k, l, imageBands, imageToBands;

	imageBands = imageFrom.GetBands();
	width = imageFrom.GetWidth();
	height = imageFrom.GetHeight();
	imageToBands = imageTo.GetBands();
	points = width * height;

	int subWidth = (width + sampleSize - 1) / sampleSize;
	int subHeight = (height + sampleSize - 1) / sampleSize;
	int subPoints = subWidth * subHeight;

	LImage<T> xyImage(width, height, imageBands + 2);

	T *xyData = xyImage.GetData();
	double invXY = 1 / sigmaXY, invLuv = 1 / sigmaLuv;
	T *fromData = imageFrom.GetData();

	for(j = 0; j < height; j++) for(i = 0; i < width; i++, xyData += imageBands + 2, fromData += imageBands) 
	{
		xyData[0] = (T)(i * invXY), xyData[1] = (T)(j * invXY);
		for(k = 0; k < imageBands; k++) xyData[k + 2] = (T)(fromData[k] * invLuv);
	}

	double rangeMins[3], rangeMaxs[3];
	xyData = &xyImage(0, 0, 2);

	rangeMins[0] = 0, rangeMaxs[0] = width / sigmaXY;
	rangeMins[1] = 0, rangeMaxs[1] = height / sigmaXY;
	rangeMins[2] = rangeMaxs[2] = *xyData;

	for(i = 0; i < points; i++, xyData += imageBands + 2)
	{
		if(*xyData < rangeMins[2]) rangeMins[2] = *xyData;
		else if(*xyData > rangeMaxs[2]) rangeMaxs[2] = *xyData;
	}

	int *buckets, *bucketList, bucketNeighs[27], noBuckets[3];

	for(i = 0; i < 3; i++) noBuckets[i] = (int)(rangeMaxs[i] - rangeMins[i] + 3);
	buckets = new int[noBuckets[0] * noBuckets[1] * noBuckets[2]];
	for(i = 0; i < noBuckets[0] * noBuckets[1] * noBuckets[2]; i++) buckets[i] = -1;

	bucketList = new int[points];
	xyData = xyImage.GetData();
	for(i = 0; i < points; i++, xyData += imageBands + 2)
	{
		int bucketIndex = (int)(xyData[0] - rangeMins[0]) + 1 + noBuckets[0] * ((int)(xyData[1] - rangeMins[1]) + 1 + noBuckets[1] * ((int)(xyData[2] - rangeMins[2]) + 1));
		bucketList[i] = buckets[bucketIndex];
		buckets[bucketIndex] = i;
	}
    for(i = -1; i <= 1; i++) for(j = -1; j <= 1; j++) for(k = -1; k <= 1; k++)
    {
		bucketNeighs[k + 1 + 3 * (j + 1) + 9 * (i + 1)] = i + noBuckets[0] * (j + noBuckets[1] * k);
    }

	unsigned char *modeTable;
	modeTable = new unsigned char[subPoints];
	memset(modeTable, 0, subPoints);

	double point[5], meanDiff[5];
	int pointCount = 0;
	int *pointList;

	pointList = new int[subPoints];

	for(l = 0; l < subHeight; l++)
	{
		xyData = xyImage(0, l * sampleSize);
		for(i = 0; i < subWidth; i++, xyData += (imageBands + 2) * sampleSize) if(modeTable[i + l * subWidth] != 1)
		{
			for (j = 0; j < imageBands + 2; j++) point[j] = xyData[j];

			double meanAbs;
			CalculateMeanDiff(xyImage, point, meanDiff, &meanAbs, rangeMins, buckets, noBuckets, bucketNeighs, bucketList);

			int iterationCount = 1;
			while((meanAbs >= 0.01) && (iterationCount < 100))
			{
				for(j = 0; j < imageBands + 2; j++) point[j] += meanDiff[j];

				if(!skipNeighbours)
				{
					int modeCandidate = ((int) ((sigmaXY * point[1] + 0.5) + sampleSize - 1) / sampleSize) * subWidth + ((int) (sigmaXY * point[0] + 0.5) + sampleSize - 1) / sampleSize;

					if((modeTable[modeCandidate] != 2) && (modeCandidate != i))
					{
						T *modeData = imageTo.GetData() + imageToBands * modeCandidate + bandTo;
						double diff = 0;
						for(k = 2; k < 2 + imageBands; k++) diff += (modeData[k] - point[k]) * (modeData[k] - point[k]);

						if(diff < 0.5)
						{
							if(modeTable[modeCandidate] == 0)
							{
								pointList[pointCount] = modeCandidate;
								pointCount++;
								modeTable[modeCandidate] = 2;
							}
							else
							{
								for(j = 0; j < imageBands; j++) point[j + 2] = modeData[j] / sigmaLuv;
								modeTable[i] = 1;
								meanAbs = -1;
							}
						}
					}
				}
				CalculateMeanDiff(xyImage, point, meanDiff, &meanAbs, rangeMins, buckets, noBuckets, bucketNeighs, bucketList);
				iterationCount++;
			}

			if(meanAbs >= 0)
			{
				for(j = 0; j < imageBands + 2; j++) point[j] += meanDiff[j];
				modeTable[i] = 1;
			}
			for(k = 0; k < imageBands; k++) point[k + 2] *= sigmaLuv;

			if(!skipNeighbours)
			{
				for(j = 0; j < pointCount; j++)
				{
					int modeCandidate = pointList[j];
					modeTable[modeCandidate] = 1;
					T *modeData = imageTo.GetData() + imageToBands * modeCandidate + bandTo;

					for(k = 0; k < imageBands; k++) modeData[k] = (T)point[k + 2];
				}
			}
			T *modeData = &imageTo(i, l, bandTo);
			for(j = 0; j < imageBands; j++) modeData[j] = (T)point[j + 2];
		}
	}
	delete[] modeTable;
	delete[] buckets;
	delete[] bucketList;
	delete[] pointList;
}
