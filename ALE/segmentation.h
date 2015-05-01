#ifndef __segmentation
#define __segmentation

#include "image.h"
#include <stdio.h>

class LSegmentation2D
{
	public :
		const char *folder, *extension;

		LSegmentation2D(const char *setFolder = NULL, const char *setExtension = NULL);
		virtual ~LSegmentation2D() {};
		virtual void Segment(LImage<double> &image, LSegmentImage &segmentImage) = 0;
};

class LMeanShiftSegmentation2D : public LSegmentation2D
{
	private :
		class LLinkedList
		{
			public:
				int label;
				LLinkedList *next;
				LLinkedList();
				int Insert(LLinkedList *list);
		};
		double sigmaXY, sigmaLuv;
		int minRegion;
		void Connect(LImage<double> &image, LSegmentImage &segmentImage);

	public :
		LMeanShiftSegmentation2D(double meanShiftXYScaleFactor, double meanShiftLuvScaleFactor, int minRegionSize, const char *setFolder = NULL, const char *setExtension = NULL);
		void Segment(LImage<double> &image, LSegmentImage &segmentImage);
};

class LKMeansSegmentation2D : public LSegmentation2D
{
	private :
		double xyLuvRatio;
		int distance, iterations, maxDiff;
	public :
		LKMeansSegmentation2D(double setXyLuvRatio, int setDistance, int setIterations, int setMaxDiff, const char *setFolder = NULL, const char *setExtension = NULL);
		void Segment(LImage<double> &image, LSegmentImage &segmentImage);
};

class LDummySegmentation2D : public LSegmentation2D
{
	private :
	public :
		LDummySegmentation2D(const char *setFolder, const char *setExtension);
		void Segment(LImage<double> &image, LSegmentImage &segmentImage);
};

#endif