#include <math.h>
#include <string.h>

#ifndef _WIN32
#include "image.h"
#endif

#include "crf.h"
#include "IL/il.h"
#include "IL/ilu.h"
#include "IL/ilut.h"

template class LImage<unsigned char>;
template class LImage<unsigned short>;
template class LImage<unsigned int>;
template class LImage<double>;
template class LImage<int>;
template class LColourImage<unsigned char>;
template class LColourImage<double>;

template <class T>
LImage<T>::LImage()
{
	data = NULL;
	width = height = bands = 0;
}

template <class T>
LImage<T>::LImage(int setBands)
{
	data = NULL;
	width = height = 0;
	bands = setBands;
}

template <class T>
LImage<T>::LImage(int setWidth, int setHeight, int setBands)
{
	data = new T[setWidth * setHeight * setBands];
	width = setWidth, height = setHeight, bands = setBands;
}

template <class T>
LImage<T>::LImage(LImage<T> &image)
{
	width = image.GetWidth();
	height = image.GetHeight();
	bands = image.GetBands();
	data = new T[width * height * bands];
	memcpy(data, image.GetData(), width * height * bands * sizeof(T));
}

template <class T>
LImage<T>::LImage(char *fileName)
{
	data = NULL;
	Load(fileName);
}

template <class T>
LImage<T>::~LImage()
{
	if(data != NULL) delete[] data;
}

template <class T>
void LImage<T>::SetResolution(int setWidth, int setHeight)
{
	if(data != NULL) delete[] data;
	data = new T[setWidth * setHeight * bands];
	width = setWidth, height = setHeight;
}

template <class T>
void LImage<T>::SetResolution(int setWidth, int setHeight, int setBands)
{
	if(data != NULL) delete[] data;
	data = new T[setWidth * setHeight * setBands];
	width = setWidth, height = setHeight, bands = setBands;
}

template <class T>
void LImage<T>::CopyDataFrom(T *fromData)
{
	memcpy(data, fromData, width * height * bands * sizeof(T));
}

template <class T>
void LImage<T>::CopyDataTo(T *toData)
{
	memcpy(toData, data, width * height * bands * sizeof(T));
}

template <class T>
void LImage<T>::CopyDataFrom(LImage<T> &image)
{
	if(data != NULL) delete[] data;
	width = image.GetWidth();
	height = image.GetHeight();
	bands = image.GetBands();
	data = new T[width * height * bands];
	memcpy(data, image.GetData(), width * height * bands * sizeof(T));
}

template <class T>
T *LImage<T>::GetData()
{
	return(data);
}

template <class T>
T &LImage<T>::GetValue(int index)
{
	return(data[index]);
}

template <class T>
T *LImage<T>::operator()(int x, int y)
{
	return(&data[(y * width + x) * bands]);
}

template <class T>
T &LImage<T>::operator()(int x, int y, int band)
{
	return(data[(y * width + x) * bands + band]);
}

template <class T>
int LImage<T>::GetWidth()
{
	return(width);
}

template <class T>
int LImage<T>::GetHeight()
{
	return(height);
}

template <class T>
int LImage<T>::GetBands()
{
	return(bands);
}

template <class T>
int LImage<T>::GetPoints()
{
	return(width * height);
}

template <class T>
int LImage<T>::GetSize()
{
	return(width * height * bands);
}

template <class T>
void LImage<T>::WindowTo(LImage &imageTo, int centreX, int centreY, int halfWidth, int halfHeight)
{
	int i, j;
	imageTo.SetResolution(halfWidth * 2 + 1, halfHeight * 2 + 1);
	for(i = 0; i < 2 * halfHeight + 1; i++)
	{
		int posY = centreY - halfHeight + i;
		T *lineFrom = (*this)(0, (posY < 0) ? 0 : ((posY > height - 1) ? height - 1 : posY));
		T *lineTo = imageTo(0, i);

		for(j = 0; (j < halfWidth - centreX) && (j < 2 * halfWidth + 1); j++) memcpy(lineTo + j * bands, lineFrom, bands * sizeof(T));
		for(j = (width - centreX + halfWidth >= 0) ? width - centreX + halfWidth : 0; j < 2 * halfWidth + 1; j++) memcpy(lineTo + j * bands, lineFrom + (width - 1) * bands, bands * sizeof(T));
		int diff = ((halfWidth - centreX > 0) ? halfWidth - centreX : 0) + ((halfWidth + 1 > width - centreX) ? halfWidth + 1 - width + centreX : 0);
		if(2 * halfWidth + 1 > diff) memcpy(lineTo + ((halfWidth - centreX > 0) ? halfWidth - centreX : 0) * bands, lineFrom + ((centreX - halfWidth > 0) ? centreX - halfWidth : 0) * bands, (2 * halfWidth + 1 - diff) * bands * sizeof(T));
	}
}

template <class T>
void LImage<T>::WindowFrom(LImage &imageFrom, int centreX, int centreY, int halfWidth, int halfHeight)
{
	imageFrom.WindowTo(*this, centreX, centreY, halfWidth, halfHeight);
}

template <class T>
void LImage<T>::Save(char *fileName)
{
	FILE *f;
	f = fopen(fileName, "wb");
	fwrite(&width, 4, 1, f);
	fwrite(&height, 4, 1, f);
	fwrite(&bands, 4, 1, f);
	fwrite(data, sizeof(T), width * height * bands, f);
	fclose(f);
}

template <class T>
void LImage<T>::Load(char *fileName)
{
	if(data != NULL) delete[] data;

	FILE *f;
	f = fopen(fileName, "rb");
	if(f == NULL) _error(fileName);

	fread(&width, 4, 1, f);
	fread(&height, 4, 1, f);
	fread(&bands, 4, 1, f);

	data = new T[width * height * bands];
	fread(data, sizeof(T), width * height * bands, f);
	fclose(f);
}

template <class T>
int LImage<T>::Exist(char *fileName)
{
	FILE *f;
	f = fopen(fileName, "rb");
	if(f == NULL) return(0);
	else
	{
   		fclose(f);
		return(1);
	}
}

template <class T>
LColourImage<T>::LColourImage() : LImage<T>()
{
}

template <class T>
LColourImage<T>::LColourImage(int setBands) : LImage<T>(setBands)
{
}

template <class T>
LColourImage<T>::LColourImage(int setWidth, int setHeight, int setBands) : LImage<T>(setWidth, setHeight, setBands)
{
}

template <class T>
LColourImage<T>::LColourImage(LColourImage<T> &image) : LImage<T>(image)
{
}

template <class T>
void LColourImage<T>::FilterTo(LColourImage<T> &toImage, LFilter2D<T> &filter)
{
	toImage.SetResolution(this->width, this->height);
	filter.Filter(*this, 0, toImage, 0, 1);
}

template <class T>
void LColourImage<T>::FilterFrom(LColourImage<T> &fromImage, LFilter2D<T> &filter)
{
	fromImage.FilterTo(*this, filter);
}

template <class T>
void LColourImage<T>::ScaleTo(LColourImage &imageTo, double scale)
{
	LGaussianFilter2D<T> filter(scale, this->bands);
	int subWidth = (int)ceil(this->width / scale);
	int subHeight = (int)ceil(this->height / scale);
	imageTo.SetResolution(subWidth, subHeight, this->bands);
	for(int i = 0; i < subHeight; i++) for(int j = 0; j < subWidth; j++) filter.Response(*this, j * scale, i * scale, imageTo(j, i));
}

template <class T>
void LColourImage<T>::ScaleFrom(LColourImage &imageFrom, double scale)
{
	imageFrom.ScaleTo(*this, scale);
}

template <class T>
void LColourImage<T>::ScaleTo(LColourImage &imageTo, double centreX, double centreY, double scale, int halfWidth, int halfHeight)
{
	imageTo.SetResolution(halfWidth * 2 + 1, halfHeight * 2 + 1);
	LGaussianFilter2D<T> filter(scale, this->bands);
	for(int i = 0; i < halfHeight * 2 + 1; i++) for(int j = 0; j < halfWidth * 2 + 1; j++)
	filter.Response(*this, centreX + scale * (j - halfWidth), centreY + scale * (i - halfHeight), imageTo(j, i));
}

template <class T>
void LColourImage<T>::ScaleFrom(LColourImage &imageFrom, double centreX, double centreY, double scale, int halfWidth, int halfHeight)
{
	imageFrom.ScaleTo(*this, centreX, centreY, scale, halfWidth, halfHeight);
}

template <class T>
void LColourImage<T>::RotateTo(LColourImage &imageTo, double centreX, double centreY, double angle, int halfWidth, int halfHeight)
{
	imageTo.SetResolution(halfWidth * 2 + 1, halfHeight * 2 + 1);
	LGaussianFilter2D<T> filter(1.0, this->bands);
	for(int i = 0; i < halfHeight * 2 + 1; i++) for(int j = 0; j < halfWidth * 2 + 1; j++)
	filter.Response(*this, centreX + (j - halfWidth) * cos(angle) + (i - halfHeight) * sin(angle), centreY - (j - halfWidth) * sin(angle) + (i - halfHeight) * cos(angle), imageTo(j, i));
}

template <class T>
void LColourImage<T>::RotateFrom(LColourImage &imageFrom, double centreX, double centreY, double angle, int halfWidth, int halfHeight)
{
	imageFrom.RotateTo(*this, centreX, centreY, angle, halfWidth, halfHeight);
}

template <class T>
void LColourImage<T>::ScaleRotateTo(LColourImage &imageTo, double centreX, double centreY, double scale, double angle, int halfWidth, int halfHeight)
{
	imageTo.SetResolution(halfWidth * 2 + 1, halfHeight * 2 + 1);
	LGaussianFilter2D<T> filter(scale, this->bands);
	for(int i = 0; i < halfHeight * 2 + 1; i++) for(int j = 0; j < halfWidth * 2 + 1; j++)
	filter.Response(*this, centreX + scale * ((j - halfWidth) * cos(angle) + (i - halfHeight) * sin(angle)), centreY - scale * ((j - halfWidth) * sin(angle) - (i - halfHeight) * cos(angle)), imageTo(j, i));
}

template <class T>
void LColourImage<T>::ScaleRotateFrom(LColourImage &imageFrom, double centreX, double centreY, double scale, double angle, int halfWidth, int halfHeight)
{
	imageFrom.ScaleRotateTo(*this, centreX, centreY, scale, angle, halfWidth, halfHeight);
}

template <class T>
void LColourImage<T>::AffineTo(LColourImage &imageTo, double centreX, double centreY, double scale, double *u, int halfWidth, int halfHeight)
{
	imageTo.SetResolution(halfWidth * 2 + 1, halfHeight * 2 + 1);
	LGaussianFilter2D<T> filter(scale, this->bands);

	for(int i = 0; i < halfHeight * 2 + 1; i++) for(int j = 0; j < halfWidth * 2 + 1; j++)
	filter.Response(*this, centreX + scale * ((j - halfWidth) * u[0] + (i - halfHeight) * u[1]), centreY + scale * ((j - halfWidth) * u[2] + (i - halfHeight) * u[3]), imageTo(j, i));
}

template <class T>
void LColourImage<T>::AffineFrom(LColourImage &imageFrom, double centreX, double centreY, double scale, double *u, int halfWidth, int halfHeight)
{
	imageFrom.AffineTo(*this, centreX, centreY, scale, u, halfWidth, halfHeight);
}

template <class T>
void LColourImage<T>::AffineRotateTo(LColourImage &imageTo, double centreX, double centreY, double scale, double *u, double angle, int halfWidth, int halfHeight)
{
	imageTo.SetResolution(halfWidth * 2 + 1, halfHeight * 2 + 1);
	LGaussianFilter2D<T> filter(scale, this->bands);

	for(int i = 0; i < halfHeight * 2 + 1; i++) for(int j = 0; j < halfWidth * 2 + 1; j++)
	{
		double x = scale * ((j - halfWidth) * u[0] + (i - halfHeight) * u[1]);
		double y = scale * ((j - halfWidth) * u[2] + (i - halfHeight) * u[3]);
		filter.Response(*this, centreX + x * cos(angle) + y * sin(angle), centreY - x * sin(angle) + y * cos(angle), imageTo(j, i));
	}
}

template <class T>
void LColourImage<T>::AffineRotateFrom(LColourImage &imageFrom, double centreX, double centreY, double scale, double *u, double angle, int halfWidth, int halfHeight)
{
	imageFrom.AffineRotateTo(*this, centreX, centreY, scale, u, angle, halfWidth, halfHeight);
}


LRgbImage::LRgbImage() : LColourImage<unsigned char>(3)
{
}

LRgbImage::LRgbImage(int setWidth, int setHeight) : LColourImage<unsigned char>(setWidth, setHeight, 3)
{
}

LRgbImage::LRgbImage(char *fileName) : LColourImage<unsigned char>(3)
{
	Load(fileName);
}

LRgbImage::LRgbImage(LRgbImage &rgbImage) : LColourImage<unsigned char>(rgbImage)
{
}

LRgbImage::LRgbImage(LGreyImage &greyImage) : LColourImage<unsigned char>(3)
{
	greyImage.Save(*this);
}

LRgbImage::LRgbImage(LLuvImage &luvImage) : LColourImage<unsigned char>(3)
{
	luvImage.Save(*this);
}

LRgbImage::LRgbImage(LLabImage &labImage) : LColourImage<unsigned char>(3)
{
	labImage.Save(*this);
}

LRgbImage::LRgbImage(LLabelImage &labelImage, LCrfDomain *domain) : LColourImage<unsigned char>(3)
{
	labelImage.Save(*this, domain);
}

LRgbImage::LRgbImage(LLabelImage &labelImage, LDataset *dataset, void (LDataset::*newLabelToRgb)(unsigned char *, unsigned char *)) : LColourImage<unsigned char>(3)
{
	labelImage.Save(*this, dataset, newLabelToRgb);
}

LRgbImage::LRgbImage(LSegmentImage &segmentImage, LRgbImage &rgbImage, int showBoundaries) : LColourImage<unsigned char>(3)
{
	segmentImage.Save(*this, rgbImage, showBoundaries);
}

LRgbImage::LRgbImage(LCostImage &costImage, LCrfDomain *domain, int showMaximum) : LColourImage<unsigned char>(3)
{
	costImage.Save(*this, domain, showMaximum);
}

void LRgbImage::Load(char *fileName)
{
	ILuint ilImage;

	ilGenImages(1, &ilImage);
	ilBindImage(ilImage);

    ilEnable(IL_ORIGIN_SET); 
    ilOriginFunc(IL_ORIGIN_LOWER_LEFT); 

	ilLoadImage(fileName);
	ilConvertImage(IL_RGB, IL_UNSIGNED_BYTE);

	SetResolution(ilGetInteger(IL_IMAGE_WIDTH), ilGetInteger(IL_IMAGE_HEIGHT));
	CopyDataFrom(ilGetData());

	ilDeleteImages(1, &ilImage);
}

void LRgbImage::Load(LRgbImage &rgbImage)
{
	rgbImage.Save(*this);
}

void LRgbImage::Load(LGreyImage &greyImage)
{
	greyImage.Save(*this);
}

void LRgbImage::Load(LLuvImage &luvImage)
{
	luvImage.Save(*this);
}

void LRgbImage::Load(LLabImage &labImage)
{
	labImage.Save(*this);
}

void LRgbImage::Load(LLabelImage &labelImage, LCrfDomain *domain)
{
	labelImage.Save(*this, domain);
}

void LRgbImage::Load(LLabelImage &labelImage, LDataset *dataset, void (LDataset::*newLabelToRgb)(unsigned char *, unsigned char *))
{
	labelImage.Save(*this, dataset, newLabelToRgb);
}

void LRgbImage::Load(LSegmentImage &segmentImage, LRgbImage &rgbImage, int showBoundaries)
{
	segmentImage.Save(*this, rgbImage, showBoundaries);
}

void LRgbImage::Load(LCostImage &costImage, LCrfDomain *domain, int showMaximum)
{
	costImage.Save(*this, domain, showMaximum);
}

void LRgbImage::RgbToGrey(unsigned char *rgb, double *grey)
{
	double R = ((double) rgb[0]) / (double)255.0;
	double G = ((double) rgb[1]) / (double)255.0;
	double B = ((double) rgb[2]) / (double)255.0;
    double y = (double)0.212671 * R + (double)0.715160 * G + (double)0.072169 * B;

	if (y > (double)0.008856) grey[0] = (double)116.0 * pow(y, (double)1.0 / (double)3.0) - (double)16.0;
    else grey[0] = (double)903.3 * y;
}

void LRgbImage::RgbToLuv(unsigned char *rgb, double *luv)
{
	double	L, u, v;

	double R = ((double) rgb[0]) / (double)255.0;
	double G = ((double) rgb[1]) / (double)255.0;
	double B = ((double) rgb[2]) / (double)255.0;

    double x = (double)0.412453 * R + (double)0.357580 * G + (double)0.180423 * B;
    double y = (double)0.212671 * R + (double)0.715160 * G + (double)0.072169 * B;
    double z = (double)0.019334 * R + (double)0.119193 * G + (double)0.950227 * B;

	if (y > 0.008856) L = (double)116.0 * pow(y, (double)1.0 / (double)3.0) - (double)16.0;
    else L = (double)903.3 * y;

	double sum	= x + 15 * y + 3 * z;

	if(sum != 0) u = 4 * x / sum, v = 9 * y / sum;
	else u = 4.0, v = (double)9.0 / (double)15.0;

	luv[0] = L;
    luv[1] = 13 * L * (u - (double)0.19784977571475);
    luv[2] = 13 * L * (v - (double)0.46834507665248);
}

void LRgbImage::RgbToLab(unsigned char *rgb, double *lab)
{
	double R = ((double) rgb[0]) / (double)255.0;
	double G = ((double) rgb[1]) / (double)255.0;
	double B = ((double) rgb[2]) / (double)255.0;

    double X =  0.412453 * R + 0.357580 * G + 0.180423 * B;
    double Y =  0.212671 * R + 0.715160 * G + 0.072169 * B;
    double Z =  0.019334 * R + 0.119193 * G + 0.950227 * B;

    double xr = X / 0.950456, yr = Y / 1.000, zr = Z / 1.088854;
	
    if(yr > 0.008856) lab[0] = 116.0 * pow(yr, 1.0 / 3.0) - 16.0;
	else lab[0] = 903.3 * yr;

	double fxr, fyr, fzr;

    if(xr > 0.008856) fxr = pow(xr, 1.0 / 3.0);
	else fxr = 7.787 * xr + 16.0 / 116.0;

	if(yr > 0.008856) fyr = pow(yr, 1.0 / 3.0);
	else fyr = 7.787 * yr + 16.0 / 116.0;

	if(zr > 0.008856) fzr = pow(zr, 1.0 / 3.0);
	else fzr = 7.787 * zr + 16.0 / 116.0;

	lab[1] = 500.0 * (fxr - fyr);
	lab[2] = 200.0 * (fyr - fzr);
}

void LRgbImage::Save(LRgbImage &rgbImage)
{
	rgbImage.SetResolution(width, height);
	CopyDataTo(rgbImage.GetData());
}

void LRgbImage::Save(char *fileName)
{
	ILuint ilImage;

	ilGenImages(1, &ilImage);
	ilBindImage(ilImage);

	ilTexImage(width, height, 1, 3, IL_RGB, IL_UNSIGNED_BYTE, NULL);
	CopyDataTo(ilGetData());

	ilEnable(IL_FILE_OVERWRITE);
	ilSaveImage(fileName);
	ilDeleteImages(1, &ilImage);
}


void LRgbImage::Save(LGreyImage &greyImage)
{
	int points = width * height;
	unsigned char *rgbData = data;
	greyImage.SetResolution(width, height);

	double *greyData = greyImage.GetData();
	for(int i = 0; i < points; i++, rgbData += 3, greyData++) RgbToGrey(rgbData, greyData);
}

void LRgbImage::Save(LLuvImage &luvImage)
{
	int points = width * height;
	unsigned char *rgbData = data;
	luvImage.SetResolution(width, height);

	double *luvData = luvImage.GetData();
	for(int i = 0; i < points; i++, rgbData += 3, luvData += 3) RgbToLuv(rgbData, luvData);
}

void LRgbImage::Save(LLabImage &labImage)
{
	int points = width * height;
	unsigned char *rgbData = data;
	labImage.SetResolution(width, height);

	double *labData = labImage.GetData();
	for(int i = 0; i < points; i++, rgbData += 3, labData += 3) RgbToLab(rgbData, labData);
}

void LRgbImage::Save(LLabelImage &labelImage, LCrfDomain *domain)
{
	Save(labelImage, domain->dataset, domain->rgbToLabel);
}

void LRgbImage::Save(LLabelImage &labelImage, LDataset *dataset, void (LDataset::*newRgbToLabel)(unsigned char *, unsigned char *))
{
	int points = width * height;
	unsigned char *rgbData = data;
	labelImage.SetResolution(width, height);

	unsigned char *labelData = labelImage.GetData();
	for(int i = 0; i < points; i++, rgbData += 3, labelData++) (dataset->*newRgbToLabel)(rgbData, labelData);
}

LGreyImage::LGreyImage() : LColourImage<double>(1)
{
}

LGreyImage::LGreyImage(int setWidth, int setHeight) : LColourImage<double>(setWidth, setHeight, 1)
{
}

LGreyImage::LGreyImage(char *fileName) : LColourImage<double>(1)
{
	Load(fileName);
}

LGreyImage::LGreyImage(LRgbImage &rgbImage) : LColourImage<double>(1)
{
	rgbImage.Save(*this);
}

LGreyImage::LGreyImage(LGreyImage &greyImage) : LColourImage<double>(greyImage)
{
}

LGreyImage::LGreyImage(LLuvImage &luvImage) : LColourImage<double>(1)
{
	luvImage.Save(*this);
}

LGreyImage::LGreyImage(LLabImage &labImage) : LColourImage<double>(1)
{
	labImage.Save(*this);
}

void LGreyImage::Load(char *fileName)
{
	LRgbImage rgbImage(fileName);
	rgbImage.Save(*this);
}

void LGreyImage::Load(LRgbImage &rgbImage)
{
	rgbImage.Save(*this);
}

void LGreyImage::Load(LGreyImage &greyImage)
{
	greyImage.Save(*this);
}

void LGreyImage::Load(LLuvImage &luvImage)
{
	luvImage.Save(*this);
}

void LGreyImage::Load(LLabImage &labImage)
{
	labImage.Save(*this);
}

void LGreyImage::GreyToRgb(double *grey, unsigned char *rgb)
{
	if(grey[0] < (double)0.1) rgb[0] = rgb[1] = rgb[2] = 0;
	else
	{
		double	y;
		if(grey[0] <= (double)7.9996) y	= 255 * grey[0] / (double)903.3;
		else y = (grey[0] + (double)16.0) / (double)116.0, y  = 255 * y * y * y;

		rgb[0] = (unsigned char)((y < 0) ? 0 : ((y > 255) ? 255 : y));
		rgb[2] = rgb[1] = rgb[0];
	}
}

void LGreyImage::GreyToLuv(double *grey, double *luv)
{
	luv[0] = grey[0], luv[1] = luv[2] = 0;
}

void LGreyImage::GreyToLab(double *grey, double *lab)
{
	lab[0] = grey[0], lab[1] = lab[2] = 0;
}

void LGreyImage::Save(char *fileName)
{
	LRgbImage rgbImage(*this);
	rgbImage.Save(fileName);
}

void LGreyImage::Save(LRgbImage &rgbImage)
{
	int points = width * height;
	double *greyData = data;

	rgbImage.SetResolution(width, height);
	unsigned char *rgbData = rgbImage.GetData();

	for(int i = 0; i < points; i++, rgbData += 3, greyData++) GreyToRgb(greyData, rgbData);
}

void LGreyImage::Save(LGreyImage &greyImage)
{
	greyImage.SetResolution(width, height);
	CopyDataTo(greyImage.GetData());
}

void LGreyImage::Save(LLuvImage &luvImage)
{
	int points = width * height;
	double *greyData = data;

	luvImage.SetResolution(width, height);
	double *luvData = luvImage.GetData();

	for(int i = 0; i < points; i++, greyData++, luvData += 3) GreyToLuv(greyData, luvData);
}

void LGreyImage::Save(LLabImage &labImage)
{
	int points = width * height;
	double *greyData = data;

	labImage.SetResolution(width, height);
	double *labData = labImage.GetData();

	for(int i = 0; i < points; i++, greyData++, labData += 3) GreyToLab(greyData, labData);
}

LLuvImage::LLuvImage() : LColourImage<double>(3)
{
}

LLuvImage::LLuvImage(int setWidth, int setHeight) : LColourImage<double>(setWidth, setHeight, 3)
{
}

LLuvImage::LLuvImage(char *fileName) : LColourImage<double>(3)
{
	Load(fileName);
}

LLuvImage::LLuvImage(LRgbImage &rgbImage) : LColourImage<double>(3)
{
	rgbImage.Save(*this);
}

LLuvImage::LLuvImage(LGreyImage &greyImage) : LColourImage<double>(3)
{
	greyImage.Save(*this);
}

LLuvImage::LLuvImage(LLabImage &labImage) : LColourImage<double>(3)
{
	LRgbImage rgbImage(labImage);
	rgbImage.Save(*this);
}

LLuvImage::LLuvImage(LLuvImage &luvImage) : LColourImage<double>(luvImage)
{
}

void LLuvImage::Load(char *fileName)
{
	LRgbImage rgbImage(fileName);
	rgbImage.Save(*this);
}

void LLuvImage::Load(LRgbImage &rgbImage)
{
	rgbImage.Save(*this);
}

void LLuvImage::Load(LGreyImage &greyImage)
{
	greyImage.Save(*this);
}

void LLuvImage::Load(LLuvImage &luvImage)
{
	luvImage.Save(*this);
}

void LLuvImage::Load(LLabImage &labImage)
{
	LRgbImage rgbImage(labImage);
	rgbImage.Save(*this);
}

void LLuvImage::LuvToRgb(double *luv, unsigned char *rgb)
{
	if(luv[0] < 0.1) rgb[0] = rgb[1] = rgb[2] = 0;
	else
	{
		double	x, y, z, u, v;

		if(luv[0] <= (double)7.9996) y = luv[0] / (double)903.3;
		else y = (luv[0] + (double)16.0) / (double)116.0, y  = y * y * y;

		u = luv[1] / (13 * luv[0]) + (double)0.19784977571475;
		v = luv[2] / (13 * luv[0]) + (double)0.46834507665248;
		x = 9 * u * y / (4 * v);
		z = (12 - 3 * u - 20 * v) * y / (4 * v);

		double R = (double)3.240479 * x - (double)1.537150 * y - (double)0.498535 * z;
		double G = (double)-0.969256 * x + (double)1.875992 * y + (double)0.041556 * z;
		double B = (double)0.055648 * x - (double)0.204043 * y + (double)1.057311 * z;
		R *= 255, G *= 255, B *= 255;

		rgb[0] = (unsigned char)((R < 0) ? 0 : ((R > 255) ? 255 : R));
		rgb[1] = (unsigned char)((G < 0) ? 0 : ((G > 255) ? 255 : G));
		rgb[2] = (unsigned char)((B < 0) ? 0 : ((B > 255) ? 255 : B));
	}
}

void LLuvImage::LuvToGrey(double *luv, double *grey)
{
	grey[0] = luv[0];
}

void LLuvImage::Save(char *fileName)
{
	LRgbImage rgbImage(*this);
	rgbImage.Save(fileName);
}

void LLuvImage::Save(LRgbImage &rgbImage)
{
	int points = width * height;
	double *luvData = data;
	rgbImage.SetResolution(width, height);

	unsigned char *rgbData = rgbImage.GetData();
	for(int i = 0; i < points; i++, rgbData += 3, luvData += 3) LuvToRgb(luvData, rgbData);
}

void LLuvImage::Save(LGreyImage &greyImage)
{
	int points = width * height;
	greyImage.SetResolution(width, height);
	double *luvData = data;

	double *greyData = greyImage.GetData();
	for(int i = 0; i < points; i++, greyData++, luvData += 3) LuvToGrey(luvData, greyData);
}

void LLuvImage::Save(LLuvImage &luvImage)
{
	luvImage.SetResolution(width, height);
	CopyDataTo(luvImage.GetData());
}

void LLuvImage::Save(LLabImage &labImage)
{
	LRgbImage rgbImage(*this);
	rgbImage.Save(labImage);
}

LLabImage::LLabImage() : LColourImage<double>(3)
{
}

LLabImage::LLabImage(int setWidth, int setHeight) : LColourImage<double>(setWidth, setHeight, 3)
{
}

LLabImage::LLabImage(char *fileName) : LColourImage<double>(3)
{
	Load(fileName);
}

LLabImage::LLabImage(LRgbImage &rgbImage) : LColourImage<double>(3)
{
	rgbImage.Save(*this);
}

LLabImage::LLabImage(LGreyImage &greyImage) : LColourImage<double>(3)
{
	greyImage.Save(*this);
}

LLabImage::LLabImage(LLuvImage &luvImage) : LColourImage<double>(3)
{
	LRgbImage rgbImage(luvImage);
	rgbImage.Save(*this);
}

LLabImage::LLabImage(LLabImage &labImage) : LColourImage<double>(labImage)
{
}

void LLabImage::Load(char *fileName)
{
	LRgbImage rgbImage(fileName);
	rgbImage.Save(*this);
}

void LLabImage::Load(LRgbImage &rgbImage)
{
	rgbImage.Save(*this);
}

void LLabImage::Load(LGreyImage &greyImage)
{
	greyImage.Save(*this);
}

void LLabImage::Load(LLabImage &labImage)
{
	labImage.Save(*this);
}

void LLabImage::Load(LLuvImage &luvImage)
{
	LRgbImage rgbImage(luvImage);
	rgbImage.Save(*this);
}

void LLabImage::LabToRgb(double *lab, unsigned char *rgb)
{
    double X, Y, Z;
    double P = (lab[0] + 16.0) / 116.0;

	if(lab[0] > 7.9996) Y = 1.000 * P * P * P;
	else Y = 1.000 * lab[0] / 903.3;

    double yr = Y / 1.000, fy;
	if(yr > 0.008856) fy = pow(yr, 1.0 / 3.0);
	else fy = 7.787 * yr + 16.0 / 116.0;

	double fx = lab[1] / 500.0 + fy, fz = fy - lab[2] / 200.0;

	if(fx > 0.2069) X = 0.950456 * fx * fx * fx;
	else X = 0.950456 / 7.787 * (fx - 16.0 / 116.0);

	if(fz > 0.2069) Z = 1.088854 * fz * fz * fz;
	else Z = 1.088854 / 7.787 * (fz - 16.0 / 116.0);

    double R = 3.240479 * X - 1.537150 * Y - 0.498535 * Z;
    double G = -0.969256 * X + 1.875992 * Y + 0.041556 * Z;
    double B = 0.055648 * X - 0.204043 * Y + 1.057311 * Z;
	R *= 255, G *= 255, B *= 255;

	rgb[0] = (unsigned char)((R < 0) ? 0 : ((R > 255) ? 255 : R));
	rgb[1] = (unsigned char)((G < 0) ? 0 : ((G > 255) ? 255 : G));
	rgb[2] = (unsigned char)((B < 0) ? 0 : ((B > 255) ? 255 : B));
}

void LLabImage::LabToGrey(double *lab, double *grey)
{
	grey[0] = lab[0];
}

void LLabImage::Save(char *fileName)
{
	LRgbImage rgbImage(*this);
	rgbImage.Save(fileName);
}

void LLabImage::Save(LRgbImage &rgbImage)
{
	int points = width * height;
	double *labData = data;
	rgbImage.SetResolution(width, height);

	unsigned char *rgbData = rgbImage.GetData();
	for(int i = 0; i < points; i++, rgbData += 3, labData += 3) LabToRgb(labData, rgbData);
}

void LLabImage::Save(LGreyImage &greyImage)
{
	int points = width * height;
	greyImage.SetResolution(width, height);
	double *labData = data;

	double *greyData = greyImage.GetData();
	for(int i = 0; i < points; i++, greyData++, labData += 3) LabToGrey(labData, greyData);
}

void LLabImage::Save(LLuvImage &luvImage)
{
	LRgbImage rgbImage(*this);
	rgbImage.Save(luvImage);
}

void LLabImage::Save(LLabImage &labImage)
{
	labImage.SetResolution(width, height);
	CopyDataTo(labImage.GetData());
}

LLabelImage::LLabelImage() : LImage<unsigned char>(1)
{
}

LLabelImage::LLabelImage(int setWidth, int setHeight) : LImage<unsigned char>(setWidth, setHeight, 1)
{
}

LLabelImage::LLabelImage(char *fileName, LCrfDomain *domain) : LImage<unsigned char>(1)
{
	Load(fileName, domain);
}

LLabelImage::LLabelImage(char *fileName, LDataset *dataset, void (LDataset::*newRgbToLabel)(unsigned char *, unsigned char *)) : LImage<unsigned char>(1)
{
	Load(fileName, dataset, newRgbToLabel);
}

LLabelImage::LLabelImage(LRgbImage &rgbImage, LCrfDomain *domain) : LImage<unsigned char>(1)
{
	rgbImage.Save(*this, domain);
}

LLabelImage::LLabelImage(LRgbImage &rgbImage, LDataset *dataset, void (LDataset::*newRgbToLabel)(unsigned char *, unsigned char *)) : LImage<unsigned char>(1)
{
	rgbImage.Save(*this, dataset, newRgbToLabel);
}

LLabelImage::LLabelImage(LCostImage &costImage, int showMaximum) : LImage<unsigned char>(1)
{
	costImage.Save(*this, showMaximum);
}

LLabelImage::LLabelImage(LLabelImage &labelImage) : LImage<unsigned char>(labelImage)
{
}

void LLabelImage::Save(char *fileName, LRgbImage &rgbImage, LCrfDomain *domain)
{
	LGreyImage greyImage(rgbImage);
	LRgbImage labelRgbImage(*this, domain);

	unsigned char *rgbData = labelRgbImage.GetData();
	double *greyData = greyImage.GetData();

	for(int i = 0; i < width * height; i++, rgbData += 3, greyData++)
	{
		rgbData[0] = (unsigned char)(rgbData[0] * *greyData / 100);
		rgbData[1] = (unsigned char)(rgbData[1] * *greyData / 100);
		rgbData[2] = (unsigned char)(rgbData[2] * *greyData / 100);
	}
	labelRgbImage.Save(fileName);
}

void LLabelImage::Save(char *fileName, LRgbImage &rgbImage, LDataset *dataset, void (LDataset::*newLabelToRgb)(unsigned char *, unsigned char *))
{
	LGreyImage greyImage(rgbImage);
	LRgbImage labelRgbImage(*this, dataset, newLabelToRgb);

	unsigned char *rgbData = labelRgbImage.GetData();
	double *greyData = greyImage.GetData();

	for(int i = 0; i < width * height; i++, rgbData += 3, greyData++)
	{
		rgbData[0] = (unsigned char)(rgbData[0] * *greyData / 100);
		rgbData[1] = (unsigned char)(rgbData[1] * *greyData / 100);
		rgbData[2] = (unsigned char)(rgbData[2] * *greyData / 100);
	}
	labelRgbImage.Save(fileName);
}

void LLabelImage::Load(char *fileName, LCrfDomain *domain)
{
	LRgbImage rgbImage(fileName);
	rgbImage.Save(*this, domain);
}

void LLabelImage::Load(char *fileName, LDataset *dataset, void (LDataset::*newRgbToLabel)(unsigned char *, unsigned char *))
{
	LRgbImage rgbImage(fileName);
	rgbImage.Save(*this, dataset, newRgbToLabel);
}

void LLabelImage::Load(LRgbImage &rgbImage, LCrfDomain *domain)
{
	rgbImage.Save(*this, domain);
}

void LLabelImage::Load(LRgbImage &rgbImage, LDataset *dataset, void (LDataset::*newRgbToLabel)(unsigned char *, unsigned char *))
{
	rgbImage.Save(*this, dataset, newRgbToLabel);
}

void LLabelImage::Load(LLabelImage &labelImage)
{
	labelImage.Save(*this);
}

void LLabelImage::Load(LCostImage &costImage, int showMaximum)
{
	costImage.Save(*this, showMaximum);
}

void LLabelImage::Save(char *fileName, LCrfDomain *domain)
{
	LRgbImage rgbImage(*this, domain);
	rgbImage.Save(fileName);
}

void LLabelImage::Save(char *fileName, LDataset *dataset, void (LDataset::*newLabelToRgb)(unsigned char *, unsigned char *))
{
	LRgbImage rgbImage(*this, dataset, newLabelToRgb);
	rgbImage.Save(fileName);
}

void LLabelImage::Save8bit(char *fileName)
{
	ILuint ilImage;

	unsigned char *pal = new unsigned char[3 * 256], *palData = pal;

	for(int j = 0; j < 256; j++, palData += 3)
	{
		int lab = j;
		palData[0] = palData[1] = palData[2] = 0;

		for(int i = 0; lab > 0; i++, lab >>= 3)
		{
			palData[0] |= (unsigned char) (((lab >> 0) & 1) << (7 - i));
			palData[1] |= (unsigned char) (((lab >> 1) & 1) << (7 - i));
			palData[2] |= (unsigned char) (((lab >> 2) & 1) << (7 - i));
		}
	}
	ilGenImages(1, &ilImage);
	ilBindImage(ilImage);

	ilTexImage(width, height, 1, 1, IL_COLOUR_INDEX, IL_UNSIGNED_BYTE, NULL);
	ilRegisterPal(pal, 3 * 256, IL_PAL_RGB24);

	CopyDataTo(ilGetData());

	ilEnable(IL_FILE_OVERWRITE);
	ilSaveImage(fileName);

	ilDeleteImages(1, &ilImage);
	delete[] pal;
}

void LLabelImage::Save(LRgbImage &rgbImage, LCrfDomain *domain)
{
	Save(rgbImage, domain->dataset, domain->labelToRgb);
}

void LLabelImage::Save(LRgbImage &rgbImage, LDataset *dataset, void (LDataset::*newLabelToRgb)(unsigned char *, unsigned char *))
{
	int points = width * height;
	unsigned char *labelData = data;

	rgbImage.SetResolution(width, height);
	unsigned char *rgbData = rgbImage.GetData();
	for(int i = 0; i < points; i++, rgbData += 3, labelData++) (dataset->*newLabelToRgb)(labelData, rgbData);
}

void LLabelImage::Save(LLabelImage &labelImage)
{
	labelImage.SetResolution(width, height);
	CopyDataTo(labelImage.GetData());
}

LCostImage::LCostImage() : LImage<double>()
{
}

LCostImage::LCostImage(int setWidth, int setHeight, int setBands) : LImage<double>(setWidth, setHeight, setBands)
{
}

LCostImage::LCostImage(LCostImage &costImage) : LImage<double>(costImage)
{
}

void LCostImage::CostToLabel(double *cost, unsigned char *label, int showMaximum)
{
	int lab = 0, i, bands = GetBands();
	if(showMaximum) { for(i = 1; i < bands; i++) if(cost[i] > cost[lab]) lab = i; }
	else { for(i = 1; i < bands; i++) if(cost[i] < cost[lab]) lab = i; }
	label[0] = lab + 1;
}

void LCostImage::Save(char *fileName, LCrfDomain *domain, int showMaximum)
{
	LLabelImage labelImage;
	Save(labelImage, showMaximum);
	labelImage.Save(fileName, domain);
}

void LCostImage::Save(LRgbImage &rgbImage, LCrfDomain *domain, int showMaximum)
{
	LLabelImage labelImage;
	Save(labelImage, showMaximum);
	labelImage.Save(rgbImage, domain);
}

void LCostImage::Save(LLabelImage &labelImage, int showMaximum)
{
	int points = width * height;
	double *costData = data;
	labelImage.SetResolution(width, height);
	unsigned char *labelData = labelImage.GetData();
	for(int i = 0; i < points; i++, costData += bands, labelData++) CostToLabel(costData, labelData, showMaximum);
}

LSegmentImage::LSegmentImage() : LImage<int>(1)
{
}

LSegmentImage::LSegmentImage(int setWidth, int setHeight) : LImage<int>(setWidth, setHeight, 1)
{
}

LSegmentImage::LSegmentImage(LSegmentImage &segmentImage) : LImage<int>(segmentImage)
{
}

void LSegmentImage::Save(char *fileName, LRgbImage &rgbImage, int showBoundaries)
{
	LRgbImage segmentRgbImage;
	Save(segmentRgbImage, rgbImage, showBoundaries);
	segmentRgbImage.Save(fileName);
}

void LSegmentImage::Save(LRgbImage &segmentRgbImage, LRgbImage &rgbImage, int showBoundaries)
{
	int points = width * height;
	int *segmentData = data;
	int segments = 0;
	int i, j, k;
	int *segmentRgb, *counts;
	unsigned char *rgbData = rgbImage.GetData();

	for(i = 0; i < points; i++) if(segmentData[i] + 1 > segments) segments = segmentData[i] + 1;

	segmentRgbImage.SetResolution(width, height);

	counts = new int[segments];
	segmentRgb = new int[segments * 3];
	memset(counts, 0, segments * sizeof(int));
	memset(segmentRgb, 0, 3 * segments * sizeof(int));

	for(i = 0; i < points; i++, rgbData += 3, segmentData++)
	{
		for(j = 0; j < 3; j++) segmentRgb[3 * *segmentData + j] += rgbData[j];
		counts[*segmentData]++;
	}
	for(i = 0; i < segments; i++) for(j = 0; j < 3; j++)
	{
		if(counts[i] > 0) segmentRgb[3 * i + j] /= counts[i];
		segmentRgb[3 * i + j] = (segmentRgb[3 * i + j] < 0) ? 0 : ((segmentRgb[3 * i + j] > 255) ? 255 : segmentRgb[3 * i + j]);
	}

	unsigned char *segmentRgbData = segmentRgbImage.GetData();
	segmentData = data;

	for(k = 0; k < height; k++) for(i = 0; i < width; i++, segmentRgbData += 3, segmentData++)
	{
		if((showBoundaries) && (((k > 0) && ((*this)(i, k - 1, 0) != *segmentData)) || ((k < height - 1) && ((*this)(i, k + 1, 0) != *segmentData)) || ((i > 0) && ((*this)(i - 1, k, 0) != *segmentData)) || ((i < width - 1) && ((*this)(i + 1, k, 0) != *segmentData)))) for(j = 0; j < 3; j++) segmentRgbData[j] = (unsigned char)255;
		else for(j = 0; j < 3; j++) segmentRgbData[j] = (unsigned char) segmentRgb[3 * *segmentData + j];
	}
	delete[] counts;
	delete[] segmentRgb;
}

LIntegralImage::LIntegralImage()
{
	width = height = 0;
}

int LIntegralImage4B::Response(int x1, int y1, int x2, int y2)
{
	if((x2 <= 0) || (y2 <= 0) || (x1 >= width) || (y1 >= height)) return(0);

	int maxX = (width < x2) ? width - 1 : x2 - 1;
	int maxY = (height < y2) ? height - 1 : y2 - 1;

    int r = image(maxX, maxY, 0);
    int rx = (x1 > 0) ? (image(x1 - 1, maxY, 0)) : 0;
    int ry = (y1 > 0) ? (image(maxX, y1 - 1, 0)) : 0;
    int rxy = ((x1 > 0) && (y1 > 0)) ? (image(x1 - 1, y1 - 1, 0)) : 0;

	return(r - rx - ry + rxy);
}

double LIntegralImage4B::DResponse(double x1, double y1, double x2, double y2)
{
	if((x2 <= 0) || (y2 <= 0) || (x1 >= width) || (y1 >= height)) return(0);

	double maxX = (width < x2) ? width - 1 : x2 - 1;
	double maxY = (height < y2) ? height - 1 : y2 - 1;
	double minX = (x1 > 1) ? x1 - 1 : -1;
	double minY = (y1 > 1) ? y1 - 1 : -1;

	int iminX = (int)minX, iminY = (int)minY, imaxX = (int)maxX, imaxY = (int)maxY;

	double r = (1 + imaxX - maxX) * (1 + imaxY - maxY) * image(imaxX, imaxY, 0) + ((imaxX + 1 < width) ? (maxX - imaxX) * (1 + imaxY - maxY) * image(imaxX + 1, imaxY, 0) : 0) + ((imaxY + 1 < height) ? (1 + imaxX - maxX) * (maxY - imaxY) * image(imaxX, imaxY + 1, 0) : 0) + (((imaxX + 1 < width) && (imaxY + 1 < height)) ? (maxX - imaxX) * (maxY - imaxY) * image(imaxX + 1, imaxY + 1, 0) : 0);
	double rx = (iminX >= 0) ? ((1 + iminX - minX) * (1 + imaxY - maxY) * image(iminX, imaxY, 0) + ((iminX + 1 < width) ? (minX - iminX) * (1 + imaxY - maxY) * image(iminX + 1, imaxY, 0) : 0) + ((imaxY + 1 < height) ? (1 + iminX - minX) * (maxY - imaxY) * image(iminX, imaxY + 1, 0) : 0) + (((iminX + 1 < width) && (imaxY + 1 < height)) ? (minX - iminX) * (maxY - imaxY) * image(iminX + 1, imaxY + 1, 0) : 0)) : 0;
    double ry = (iminY >= 0) ? ((1 + imaxX - maxX) * (1 + iminY - minY) * image(imaxX, iminY, 0) + ((imaxX + 1 < width) ? (maxX - imaxX) * (1 + iminY - minY) * image(imaxX + 1, iminY, 0) : 0) + ((iminY + 1 < height) ? (1 + imaxX - maxX) * (minY - iminY) * image(imaxX, iminY + 1, 0) : 0) + (((imaxX + 1 < width) && (iminY + 1 < height)) ? (maxX - imaxX) * (minY - iminY) * image(imaxX + 1, iminY + 1, 0) : 0)) : 0;
    double rxy = ((iminX >= 0) && (iminY >= 0)) ? ((1 + iminX - minX) * (1 + iminY - minY) * image(iminX, iminY, 0) + ((iminX + 1 < width) ? (minX - iminX) * (1 + iminY - minY) * image(iminX + 1, iminY, 0) : 0) + ((iminY + 1 < height) ? (1 + iminX - minX) * (minY - iminY) * image(iminX, iminY + 1, 0) : 0) + (((iminX + 1 < width) && (iminY + 1 < height)) ? (minX - iminX) * (minY - iminY) * image(iminX + 1, iminY + 1, 0) : 0)) : 0;
	
	return(r - rx - ry + rxy);
}

void LIntegralImage4B::Load(LImage<unsigned short> &dataImage, int subSample, int index)
{
	int l, k;
	int dataWidth = dataImage.GetWidth(), dataHeight = dataImage.GetHeight();
	int subWidth = (dataWidth + subSample - 1) / subSample;
    int subHeight = (dataHeight + subSample - 1) / subSample;

	width = subWidth, height = subHeight;
	image.SetResolution(subWidth, subHeight, 1);

	memset(image.GetData(), 0, subWidth * subHeight * sizeof(unsigned int));

	unsigned short *dataFrom = dataImage.GetData();
	unsigned int *dataTo = image.GetData();

	int bands = dataImage.GetBands();
	for(k = 0; k < dataHeight; k++) for(l = 0; l < dataWidth * bands; l++, dataFrom++) if(*dataFrom == index) dataTo[(k / subSample) * subWidth + (l / (subSample * bands))]++;

	for(k = 0; k < subHeight; k++) for(l = 0; l < subWidth; l++, dataTo++) (*dataTo) += ((l > 0) ? *(dataTo - 1) : 0) + ((k > 0) ? *(dataTo - subWidth) : 0) - (((l > 0) && (k > 0)) ? *(dataTo - subWidth - 1) : 0);
}

void LIntegralImage4B::Copy(LImage<double> &dataImage, int subSample, int index, double scale)
{
	int l, k;
	int dataWidth = dataImage.GetWidth(), dataHeight = dataImage.GetHeight();
	int subWidth = (dataWidth + subSample - 1) / subSample;
    int subHeight = (dataHeight + subSample - 1) / subSample;
	int bands = dataImage.GetBands();

	width = subWidth, height = subHeight;
	image.SetResolution(subWidth, subHeight, 1);

	memset(image.GetData(), 0, subWidth * subHeight * sizeof(unsigned int));

	double *dataFrom = dataImage.GetData() + index;
	unsigned int *dataTo = image.GetData();

	for(k = 0; k < dataHeight; k++) for(l = 0; l < dataWidth; l++, dataFrom += bands) dataTo[(k / subSample) * subWidth + (l / subSample)] += (unsigned int)((*dataFrom) * scale);
	for(k = 0; k < subHeight; k++) for(l = 0; l < subWidth; l++, dataTo++) (*dataTo) += ((l > 0) ? *(dataTo - 1) : 0) + ((k > 0) ? *(dataTo - subWidth) : 0) - (((l > 0) && (k > 0)) ? *(dataTo - subWidth - 1) : 0);
}

int LIntegralImage2B::Response(int x1, int y1, int x2, int y2)
{
	if((x2 <= 0) || (y2 <= 0) || (x1 >= width) || (y1 >= height)) return(0);

	int maxX = (width < x2) ? width - 1 : x2 - 1;
	int maxY = (height < y2) ? height - 1 : y2 - 1;

    int r = image(maxX, maxY, 0);
    int rx = (x1 > 0) ? (image(x1 - 1, maxY, 0)) : 0;
    int ry = (y1 > 0) ? (image(maxX, y1 - 1, 0)) : 0;
    int rxy = ((x1 > 0) && (y1 > 0)) ? (image(x1 - 1, y1 - 1, 0)) : 0;

	return(r - rx - ry + rxy);
}

double LIntegralImage2B::DResponse(double x1, double y1, double x2, double y2)
{
	if((x2 <= 0) || (y2 <= 0) || (x1 >= width) || (y1 >= height)) return(0);

	double maxX = (width < x2) ? width - 1 : x2 - 1;
	double maxY = (height < y2) ? height - 1 : y2 - 1;
	double minX = (x1 > 1) ? x1 - 1 : -1;
	double minY = (y1 > 1) ? y1 - 1 : -1;

	int iminX = (int)minX, iminY = (int)minY, imaxX = (int)maxX, imaxY = (int)maxY;

	double r = (1 + imaxX - maxX) * (1 + imaxY - maxY) * image(imaxX, imaxY, 0) + ((imaxX + 1 < width) ? (maxX - imaxX) * (1 + imaxY - maxY) * image(imaxX + 1, imaxY, 0) : 0) + ((imaxY + 1 < height) ? (1 + imaxX - maxX) * (maxY - imaxY) * image(imaxX, imaxY + 1, 0) : 0) + (((imaxX + 1 < width) && (imaxY + 1 < height)) ? (maxX - imaxX) * (maxY - imaxY) * image(imaxX + 1, imaxY + 1, 0) : 0);
	double rx = (iminX >= 0) ? ((1 + iminX - minX) * (1 + imaxY - maxY) * image(iminX, imaxY, 0) + ((iminX + 1 < width) ? (minX - iminX) * (1 + imaxY - maxY) * image(iminX + 1, imaxY, 0) : 0) + ((imaxY + 1 < height) ? (1 + iminX - minX) * (maxY - imaxY) * image(iminX, imaxY + 1, 0) : 0) + (((iminX + 1 < width) && (imaxY + 1 < height)) ? (minX - iminX) * (maxY - imaxY) * image(iminX + 1, imaxY + 1, 0) : 0)) : 0;
    double ry = (iminY >= 0) ? ((1 + imaxX - maxX) * (1 + iminY - minY) * image(imaxX, iminY, 0) + ((imaxX + 1 < width) ? (maxX - imaxX) * (1 + iminY - minY) * image(imaxX + 1, iminY, 0) : 0) + ((iminY + 1 < height) ? (1 + imaxX - maxX) * (minY - iminY) * image(imaxX, iminY + 1, 0) : 0) + (((imaxX + 1 < width) && (iminY + 1 < height)) ? (maxX - imaxX) * (minY - iminY) * image(imaxX + 1, iminY + 1, 0) : 0)) : 0;
    double rxy = ((iminX >= 0) && (iminY >= 0)) ? ((1 + iminX - minX) * (1 + iminY - minY) * image(iminX, iminY, 0) + ((iminX + 1 < width) ? (minX - iminX) * (1 + iminY - minY) * image(iminX + 1, iminY, 0) : 0) + ((iminY + 1 < height) ? (1 + iminX - minX) * (minY - iminY) * image(iminX, iminY + 1, 0) : 0) + (((iminX + 1 < width) && (iminY + 1 < height)) ? (minX - iminX) * (minY - iminY) * image(iminX + 1, iminY + 1, 0) : 0)) : 0;
	
	return(r - rx - ry + rxy);
}

void LIntegralImage2B::Load(LImage<unsigned short> &dataImage, int subSample, int index)
{
	int l, k;
	int dataWidth = dataImage.GetWidth(), dataHeight = dataImage.GetHeight();
	int subWidth = (dataWidth + subSample - 1) / subSample;
    int subHeight = (dataHeight + subSample - 1) / subSample;

	width = subWidth, height = subHeight;
	image.SetResolution(subWidth, subHeight, 1);

	memset(image.GetData(), 0, subWidth * subHeight * sizeof(unsigned short));

	unsigned short *dataFrom = dataImage.GetData();
	unsigned short *dataTo = image.GetData();
	int bands = dataImage.GetBands();

	for(k = 0; k < dataHeight; k++) for(l = 0; l < dataWidth * bands; l++, dataFrom++) if(*dataFrom == index) dataTo[(k / subSample) * subWidth + (l / (subSample * bands))]++;
	for(k = 0; k < subHeight; k++) for(l = 0; l < subWidth; l++, dataTo++) (*dataTo) += ((l > 0) ? *(dataTo - 1) : 0) + ((k > 0) ? *(dataTo - subWidth) : 0) - (((l > 0) && (k > 0)) ? *(dataTo - subWidth - 1) : 0);
}

void LIntegralImage2B::Copy(LImage<double> &dataImage, int subSample, int index, double scale)
{
	int l, k;
	int dataWidth = dataImage.GetWidth(), dataHeight = dataImage.GetHeight();
	int subWidth = (dataWidth + subSample - 1) / subSample;
    int subHeight = (dataHeight + subSample - 1) / subSample;
	int bands = dataImage.GetBands();

	width = subWidth, height = subHeight;
	image.SetResolution(subWidth, subHeight, 1);

	memset(image.GetData(), 0, subWidth * subHeight * sizeof(unsigned short));

	double *dataFrom = dataImage.GetData() + index;
	unsigned short *dataTo = image.GetData();

	for(k = 0; k < dataHeight; k++) for(l = 0; l < dataWidth; l++, dataFrom += bands) dataTo[(k / subSample) * subWidth + (l / subSample)] += (unsigned short)((*dataFrom) * scale);
	for(k = 0; k < subHeight; k++) for(l = 0; l < subWidth; l++, dataTo++) (*dataTo) += ((l > 0) ? *(dataTo - 1) : 0) + ((k > 0) ? *(dataTo - subWidth) : 0) - (((l > 0) && (k > 0)) ? *(dataTo - subWidth - 1) : 0);
}

int LIntegralImage1B::Response(int x1, int y1, int x2, int y2)
{
	if((x2 <= 0) || (y2 <= 0) || (x1 >= width) || (y1 >= height)) return(0);

	int maxX = (width < x2) ? width - 1 : x2 - 1;
	int maxY = (height < y2) ? height - 1 : y2 - 1;

    int r = image(maxX, maxY, 0);
    int rx = (x1 > 0) ? (image(x1 - 1, maxY, 0)) : 0;
    int ry = (y1 > 0) ? (image(maxX, y1 - 1, 0)) : 0;
    int rxy = ((x1 > 0) && (y1 > 0)) ? (image(x1 - 1, y1 - 1, 0)) : 0;

    return(r - rx - ry + rxy);
}

double LIntegralImage1B::DResponse(double x1, double y1, double x2, double y2)
{
	if((x2 <= 0) || (y2 <= 0) || (x1 >= width) || (y1 >= height)) return(0);

	double maxX = (width < x2) ? width - 1 : x2 - 1;
	double maxY = (height < y2) ? height - 1 : y2 - 1;
	double minX = (x1 > 1) ? x1 - 1 : -1;
	double minY = (y1 > 1) ? y1 - 1 : -1;

	int iminX = (int)minX, iminY = (int)minY, imaxX = (int)maxX, imaxY = (int)maxY;

	double r = (1 + imaxX - maxX) * (1 + imaxY - maxY) * image(imaxX, imaxY, 0) + ((imaxX + 1 < width) ? (maxX - imaxX) * (1 + imaxY - maxY) * image(imaxX + 1, imaxY, 0) : 0) + ((imaxY + 1 < height) ? (1 + imaxX - maxX) * (maxY - imaxY) * image(imaxX, imaxY + 1, 0) : 0) + (((imaxX + 1 < width) && (imaxY + 1 < height)) ? (maxX - imaxX) * (maxY - imaxY) * image(imaxX + 1, imaxY + 1, 0) : 0);
	double rx = (iminX >= 0) ? ((1 + iminX - minX) * (1 + imaxY - maxY) * image(iminX, imaxY, 0) + ((iminX + 1 < width) ? (minX - iminX) * (1 + imaxY - maxY) * image(iminX + 1, imaxY, 0) : 0) + ((imaxY + 1 < height) ? (1 + iminX - minX) * (maxY - imaxY) * image(iminX, imaxY + 1, 0) : 0) + (((iminX + 1 < width) && (imaxY + 1 < height)) ? (minX - iminX) * (maxY - imaxY) * image(iminX + 1, imaxY + 1, 0) : 0)) : 0;
    double ry = (iminY >= 0) ? ((1 + imaxX - maxX) * (1 + iminY - minY) * image(imaxX, iminY, 0) + ((imaxX + 1 < width) ? (maxX - imaxX) * (1 + iminY - minY) * image(imaxX + 1, iminY, 0) : 0) + ((iminY + 1 < height) ? (1 + imaxX - maxX) * (minY - iminY) * image(imaxX, iminY + 1, 0) : 0) + (((imaxX + 1 < width) && (iminY + 1 < height)) ? (maxX - imaxX) * (minY - iminY) * image(imaxX + 1, iminY + 1, 0) : 0)) : 0;
    double rxy = ((iminX >= 0) && (iminY >= 0)) ? ((1 + iminX - minX) * (1 + iminY - minY) * image(iminX, iminY, 0) + ((iminX + 1 < width) ? (minX - iminX) * (1 + iminY - minY) * image(iminX + 1, iminY, 0) : 0) + ((iminY + 1 < height) ? (1 + iminX - minX) * (minY - iminY) * image(iminX, iminY + 1, 0) : 0) + (((iminX + 1 < width) && (iminY + 1 < height)) ? (minX - iminX) * (minY - iminY) * image(iminX + 1, iminY + 1, 0) : 0)) : 0;
	
	return(r - rx - ry + rxy);
}

void LIntegralImage1B::Load(LImage<unsigned short> &dataImage, int subSample, int index)
{
	int l, k;
	int dataWidth = dataImage.GetWidth(), dataHeight = dataImage.GetHeight();
	int subWidth = (dataWidth + subSample - 1) / subSample;
    int subHeight = (dataHeight + subSample - 1) / subSample;

	width = subWidth, height = subHeight;
	image.SetResolution(subWidth, subHeight, 1);

	memset(image.GetData(), 0, subWidth * subHeight * sizeof(unsigned char));

	unsigned short *dataFrom = dataImage.GetData();
	unsigned char *dataTo = image.GetData();

	int bands = dataImage.GetBands();

	for(k = 0; k < dataHeight; k++) for(l = 0; l < dataWidth * bands; l++, dataFrom++) if(*dataFrom == index) dataTo[(k / subSample) * subWidth + (l / (subSample * bands))]++;
	for(k = 0; k < subHeight; k++) for(l = 0; l < subWidth; l++, dataTo++) (*dataTo) += ((l > 0) ? *(dataTo - 1) : 0) + ((k > 0) ? *(dataTo - subWidth) : 0) - (((l > 0) && (k > 0)) ? *(dataTo - subWidth - 1) : 0);
}

void LIntegralImage1B::Copy(LImage<double> &dataImage, int subSample, int index, double scale)
{
	int l, k;
	int dataWidth = dataImage.GetWidth(), dataHeight = dataImage.GetHeight();
	int subWidth = (dataWidth + subSample - 1) / subSample;
    int subHeight = (dataHeight + subSample - 1) / subSample;
	int bands = dataImage.GetBands();

	width = subWidth, height = subHeight;
	image.SetResolution(subWidth, subHeight, 1);

	memset(image.GetData(), 0, subWidth * subHeight * sizeof(unsigned char));

	double *dataFrom = dataImage.GetData() + index;
	unsigned char *dataTo = image.GetData();

	for(k = 0; k < dataHeight; k++) for(l = 0; l < dataWidth; l++, dataFrom += bands) dataTo[(k / subSample) * subWidth + (l / subSample)] += (unsigned char)((*dataFrom) * scale);
	for(k = 0; k < subHeight; k++) for(l = 0; l < subWidth; l++, dataTo++) (*dataTo) += ((l > 0) ? *(dataTo - 1) : 0) + ((k > 0) ? *(dataTo - subWidth) : 0) - (((l > 0) && (k > 0)) ? *(dataTo - subWidth - 1) : 0);
}

int LIntegralImageHB::Response(int x1, int y1, int x2, int y2)
{
	if((x2 <= 0) || (y2 <= 0) || (x1 >= width) || (y1 >= height)) return(0);

	int maxX = (width < x2) ? width - 1 : x2 - 1;
	int maxY = (height < y2) ? height - 1 : y2 - 1;

	int r = (maxX & 1) ? (image(maxX >> 1, maxY, 0) >> 4) : (image(maxX >> 1, maxY, 0) & 15);
	int rx = (x1 > 0) ? (((x1 - 1) & 1) ? (image((x1 - 1) >> 1, maxY, 0) >> 4) : (image((x1 - 1) >> 1, maxY, 0) & 15)) : 0;
    int ry = (y1 > 0) ? ((maxX & 1) ? (image(maxX >> 1, y1 - 1, 0) >> 4) : (image(maxX >> 1, y1 - 1, 0) & 15)) : 0;
	int rxy = ((x1 > 0) && (y1 > 0)) ? (((x1 - 1) & 1) ? (image((x1 - 1) >> 1, y1 - 1, 0) >> 4) : (image((x1 - 1) >> 1, y1 - 1, 0) & 15)) : 0;

    return(r - rx - ry + rxy);
}

double LIntegralImageHB::DResponse(double x1, double y1, double x2, double y2)
{
	if((x2 <= 0) || (y2 <= 0) || (x1 >= width) || (y1 >= height)) return(0);

	double maxX = (width < x2) ? width - 1 : x2 - 1;
	double maxY = (height < y2) ? height - 1 : y2 - 1;
	double minX = (x1 > 1) ? x1 - 1 : -1;
	double minY = (y1 > 1) ? y1 - 1 : -1;

	int iminX = (int)minX, iminY = (int)minY, imaxX = (int)maxX, imaxY = (int)maxY;

	double r = (1 + imaxX - maxX) * (1 + imaxY - maxY) * ((imaxX & 1) ? (image(imaxX >> 1, imaxY, 0) >> 4) : (image(imaxX >> 1, imaxY, 0) & 15)) + ((imaxX + 1 < width) ? (maxX - imaxX) * (1 + imaxY - maxY) * (((imaxX + 1) & 1) ? (image((imaxX + 1) >> 1, imaxY, 0) >> 4) : (image((imaxX + 1) >> 1, imaxY, 0) & 15)) : 0) + ((imaxY + 1 < height) ? (1 + imaxX - maxX) * (maxY - imaxY) * ((imaxX & 1) ? (image(imaxX >> 1, imaxY + 1, 0) >> 4) : (image(imaxX >> 1, imaxY + 1, 0) & 15)) : 0) + (((imaxX + 1 < width) && (imaxY + 1 < height)) ? (maxX - imaxX) * (maxY - imaxY) * (((imaxX + 1) & 1) ? (image((imaxX + 1) >> 1, imaxY + 1, 0) >> 4) : (image((imaxX + 1) >> 1, imaxY + 1, 0) & 15)) : 0);
	double rx = (iminX >= 0) ? ((1 + iminX - minX) * (1 + imaxY - maxY) * ((iminX & 1) ? (image(iminX >> 1, imaxY, 0) >> 4) : (image(iminX >> 1, imaxY, 0) & 15)) + ((iminX + 1 < width) ? (minX - iminX) * (1 + imaxY - maxY) * (((iminX + 1) & 1) ? (image((iminX + 1) >> 1, imaxY, 0) >> 4) : (image((iminX + 1) >> 1, imaxY, 0) & 15)) : 0) + ((imaxY + 1 < height) ? (1 + iminX - minX) * (maxY - imaxY) * ((iminX & 1) ? (image(iminX >> 1, imaxY + 1, 0) >> 4) : (image(iminX >> 1, imaxY + 1, 0) & 15)) : 0) + (((iminX + 1 < width) && (imaxY + 1 < height)) ? (minX - iminX) * (maxY - imaxY) * (((iminX + 1) & 1) ? (image((iminX + 1) >> 1, imaxY + 1, 0) >> 4) : (image((iminX + 1) >> 1, imaxY + 1, 0) & 15)) : 0)) : 0;
    double ry = (iminY >= 0) ? ((1 + imaxX - maxX) * (1 + iminY - minY) * ((imaxX & 1) ? (image(imaxX >> 1, iminY, 0) >> 4) : (image(imaxX >> 1, iminY, 0) & 15)) + ((imaxX + 1 < width) ? (maxX - imaxX) * (1 + iminY - minY) * (((imaxX + 1) & 1) ? (image((imaxX + 1) >> 1, iminY, 0) >> 4) : (image((imaxX + 1) >> 1, iminY, 0) & 15)) : 0) + ((iminY + 1 < height) ? (1 + imaxX - maxX) * (minY - iminY) * ((imaxX & 1) ? (image(imaxX >> 1, iminY + 1, 0) >> 4) : (image(imaxX >> 1, iminY + 1, 0) & 15)) : 0) + (((imaxX + 1 < width) && (iminY + 1 < height)) ? (maxX - imaxX) * (minY - iminY) * (((imaxX + 1) & 1) ? (image((imaxX + 1) >> 1, iminY + 1, 0) >> 4) : (image((imaxX + 1) >> 1, iminY + 1, 0) & 15)) : 0)) : 0;
    double rxy = ((iminX >= 0) && (iminY >= 0)) ? ((1 + iminX - minX) * (1 + iminY - minY) * ((iminX & 1) ? (image(iminX >> 1, iminY, 0) >> 4) : (image(iminX >> 1, iminY, 0) & 15)) + ((iminX + 1 < width) ? (minX - iminX) * (1 + iminY - minY) * (((iminX + 1) & 1) ? (image((iminX + 1) >> 1, iminY, 0) >> 4) : (image((iminX + 1) >> 1, iminY, 0) & 15)) : 0) + ((iminY + 1 < height) ? (1 + iminX - minX) * (minY - iminY) * ((iminX & 1) ? (image(iminX >> 1, iminY + 1, 0) >> 4) : (image(iminX >> 1, iminY + 1, 0) & 15)) : 0) + (((iminX + 1 < width) && (iminY + 1 < height)) ? (minX - iminX) * (minY - iminY) * (((iminX + 1) & 1) ? (image((iminX + 1) >> 1, iminY + 1, 0) >> 4) : (image((iminX + 1) >> 1, iminY + 1, 0) & 15)) : 0)) : 0;
	
	return(r - rx - ry + rxy);
}

void LIntegralImageHB::Load(LImage<unsigned short> &dataImage, int subSample, int index)
{
	int l, k;
	int dataWidth = dataImage.GetWidth(), dataHeight = dataImage.GetHeight();
	int subWidth = (dataWidth + subSample - 1) / subSample;
    int subHeight = (dataHeight + subSample - 1) / subSample;
	int realWidth = (subWidth + 1) >> 1;

	width = subWidth, height = subHeight;
	image.SetResolution(realWidth, subHeight, 1);

	memset(image.GetData(), 0, realWidth * subHeight * sizeof(unsigned char));

	unsigned short *dataFrom = dataImage.GetData();
	unsigned char *dataTo = image.GetData();
	int bands = dataImage.GetBands();

	for(k = 0; k < dataHeight; k++) for(l = 0; l < dataWidth * bands; l++, dataFrom++) if(*dataFrom == index)
	{
		if(!((l / (subSample * bands)) & 1)) dataTo[(k / subSample) * realWidth + ((l / (subSample * bands)) >> 1)]++;
		else dataTo[(k / subSample) * realWidth + ((l / (subSample * bands)) >> 1)] += 16;
	}

	for(k = 0; k < subHeight; k++) for(l = 0; l < realWidth; l++, dataTo++)
	{
		(*dataTo) += ((l > 0) ? (*(dataTo - 1) >> 4) : 0) + ((k > 0) ? (*(dataTo - realWidth) & 15) : 0) - (((l > 0) && (k > 0)) ? (*(dataTo - realWidth - 1) >> 4) : 0);
		(*dataTo) += ((*(dataTo) & 15) +  ((k > 0) ? (*(dataTo - realWidth) >> 4) : 0) - ((k > 0) ? (*(dataTo - realWidth) & 15) : 0)) << 4;
	}
}

