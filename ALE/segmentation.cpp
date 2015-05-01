#include <stdio.h>
#include <string.h>
#include <math.h>
#include "segmentation.h"

LSegmentation2D::LSegmentation2D(const char *setFolder, const char *setExtension)
{
	folder = setFolder, extension = setExtension;
}

LMeanShiftSegmentation2D::LLinkedList::LLinkedList()
{
	label = -1;
	next = NULL;
}

int LMeanShiftSegmentation2D::LLinkedList::Insert(LLinkedList *list)
{
	if(!next)
	{
		next = list;
		list->next = NULL;
		return(0);
	}
	if(next->label > list->label)
	{
		list->next	= next;
		next = list;
		return(0);
	}

	LLinkedList	*cur = next;
	while(cur)
	{
		if(list->label == cur->label) return(1);
		else if((!(cur->next)) || (cur->next->label > list->label))
		{
			list->next = cur->next;
			cur->next = list;
			return(0);
		}
		cur = cur->next;
	}
	return(0);
}

LMeanShiftSegmentation2D::LMeanShiftSegmentation2D(double meanShiftXYScaleFactor, double meanShiftLuvScaleFactor, int minRegionSize, const char *setFolder, const char *setExtension) : LSegmentation2D(setFolder, setExtension)
{
	sigmaXY = meanShiftXYScaleFactor, sigmaLuv = meanShiftLuvScaleFactor, minRegion = minRegionSize;
}

void LMeanShiftSegmentation2D::Segment(LImage<double> &image, LSegmentImage &segmentImage)
{
	LMeanShiftFilter2D<double> filter(sigmaXY, sigmaLuv, 0, image.GetBands());
	LImage<double> output(image.GetWidth(), image.GetHeight(), image.GetBands());

	filter.Filter(image, 0, output, 0, 1);
	Connect(output, segmentImage);
}

void LMeanShiftSegmentation2D::Connect(LImage<double> &image, LSegmentImage &segmentImage)
{
	int neighX[8], neighY[8], *indexTable;
	int width, height, points, i, j, k, bands;

	width = image.GetWidth();
	height = image.GetHeight();
	bands = image.GetBands();
	points = width * height;

	segmentImage.SetResolution(width, height);

	neighX[0] = 1, neighY[0] = 0;
	neighX[1] = 1, neighY[1] = -width;
	neighX[2] = 0, neighY[2] = -width;
	neighX[3] = -1, neighY[3] = -width;
	neighX[4] = -1, neighY[4] = 0;
	neighX[5] = -1, neighY[5] = width;
	neighX[6] = 0, neighY[6] = width;
	neighX[7] = 1, neighY[7] = width;

	indexTable = new int[points];

	double *modes;
	int *modePointCounts;

	modes = new double[points * bands];
	modePointCounts = new int[points];

	int *segmentData = segmentImage.GetData();
	for(i = 0; i < points; i++, segmentData++) *segmentData = -1, modePointCounts[i] = 0;

	int label = -1;
	int regionCount, oldRegionCount;

	double *luvData = image.GetData();
	double *startLuvData = luvData;

	segmentData = segmentImage.GetData();
	int *startLabelData = segmentData;

	for(i = 0; i < points; i++, luvData += bands, segmentData++)
	{
		if(*segmentData < 0)
		{
			int	index = 0, stop = 0, startIndex = i;

			label++, *segmentData = label, indexTable[0] = startIndex;
			for(k = 0; k < bands; k++) modes[bands * label + k] = luvData[k];
			modePointCounts[label]++;

			while(!stop)
			{
				int neighFound = 0;
				int regionX = startIndex % width, regionY = startIndex / width;

				for(j = 0; j < 8; j++)
				{
					int neighIndex = startIndex + neighX[j] + neighY[j];
					if(((regionX > 0) || (neighX[j] >= 0)) && ((regionX < width - 1) || (neighX[j] <= 0)) && ((regionY > 0) || (neighY[j] >= 0)) && ((regionY < height - 1) || (neighY[j] <= 0)) && (startLabelData[neighIndex] < 0))
					{
						int notClose = 0;
						for(k = 0; (k < bands) && (!notClose); k++) if(fabs(startLuvData[startIndex * bands + k] - startLuvData[neighIndex * bands + k]) >= 1.0) notClose = 1;

						if(!notClose)
						{
							startLabelData[neighIndex] = label;
							index++, indexTable[index] = neighIndex;
							neighFound	= 1;
							modePointCounts[label]++;
						}
					}
				}
				if(neighFound) startIndex = indexTable[index];
				else if(index > 1) index--, startIndex = indexTable[index];
				else stop = 1;
			}
		}
	}
	delete[] indexTable;
	regionCount	= label + 1;

	LLinkedList *linkedList, *pool;

	linkedList = new LLinkedList[regionCount];
	pool = new LLinkedList[10 * regionCount];

	int counter = 0;
	do
	{
		for(i = 0; i < regionCount; i++) linkedList[i].label = i, linkedList[i].next = NULL;
		for(i = 0; i < 10 * regionCount - 1; i++) pool[i].next = &pool[i + 1];
		pool[10 * regionCount - 1].next = NULL;

		LLinkedList *freelinkedList	= pool, *node1, *node2, *oldFreeList;

		segmentData = segmentImage.GetData();
		for(i = 0; i < height; i++) for(j = 0; j < width; j++)
		{
			int curLabel = segmentData[i * width + j];
			if(j != width - 1)
			{
				int rightLabel = segmentData[i * width + j + 1];
				if(curLabel != rightLabel)
				{
					node1 = freelinkedList;
					node2 = freelinkedList->next;
					oldFreeList = freelinkedList;
					freelinkedList = freelinkedList->next->next;
					node1->label = curLabel;
					node2->label = rightLabel;

					linkedList[curLabel].Insert(node2);
					if(linkedList[rightLabel].Insert(node1)) freelinkedList = oldFreeList;
				}
			}
			if(i != height - 1)
			{
				int bottomLabel	= segmentData[(i + 1) * width + j];
				if(curLabel != bottomLabel)
				{
					node1 = freelinkedList;
					node2 = freelinkedList->next;
					oldFreeList = freelinkedList;
					freelinkedList = freelinkedList->next->next;
					node1->label = curLabel;
					node2->label = bottomLabel;

					linkedList[curLabel].Insert(node2);
					if(linkedList[bottomLabel].Insert(node1)) freelinkedList = oldFreeList;
				}
			}
		}
		int candidateLabel, neighCandidateLabel;
		LLinkedList *neighbour;
		for(i = 0; i < regionCount; i++)
		{
			neighbour = linkedList[i].next;

			while(neighbour)
			{
				double diff = 0;
				for(j = 0; j < bands; j++)
				{
					double el = (modes[i * bands + j] - modes[neighbour->label * bands + j]) / sigmaLuv;
					if((!j) && (modes[i * bands] > 80)) diff += 4 * el * el;
					else diff += el * el;
				}

				if(diff < 0.25)
				{
					candidateLabel = i;
					while(linkedList[candidateLabel].label != candidateLabel) candidateLabel = linkedList[candidateLabel].label;

					neighCandidateLabel	= neighbour->label;
					while(linkedList[neighCandidateLabel].label != neighCandidateLabel) neighCandidateLabel = linkedList[neighCandidateLabel].label;

					if(candidateLabel < neighCandidateLabel) linkedList[neighCandidateLabel].label = candidateLabel;
					else
					{
						linkedList[linkedList[candidateLabel].label].label = neighCandidateLabel;
						linkedList[candidateLabel].label = neighCandidateLabel;
					}
				}
				neighbour = neighbour->next;
			}
		}
		for(i = 0; i < regionCount; i++)
		{
			candidateLabel	= i;
			while(linkedList[candidateLabel].label != candidateLabel) candidateLabel = linkedList[candidateLabel].label;
			linkedList[i].label	= candidateLabel;
		}

		double *modesList = new double[bands * regionCount];
		int *pointCountList = new int[regionCount];

		for(i = 0; i < regionCount; i++) pointCountList[i] = 0;
		for(i = 0; i < bands * regionCount; i++) modesList[i] = 0;

		int thisModeCount;
		for(i = 0; i < regionCount; i++)
		{
			candidateLabel = linkedList[i].label;
			thisModeCount = modePointCounts[i];
			for(k = 0; k < bands; k++) modesList[bands * candidateLabel + k] += thisModeCount * modes[bands * i + k];
			pointCountList[candidateLabel] += thisModeCount;
		}
		int	*labelList = new int [regionCount];

		for(i = 0; i < regionCount; i++) labelList[i] = -1;

		int	label = -1;
		for(i = 0; i < regionCount; i++)
		{
			candidateLabel	= linkedList[i].label;
			if(labelList[candidateLabel] < 0)
			{
				label++;
				labelList[candidateLabel] = label;

				thisModeCount = pointCountList[candidateLabel];
				for(k = 0; k < bands; k++) modes[bands * label + k] = (modesList[bands * candidateLabel + k]) / thisModeCount;
				modePointCounts[label]	= pointCountList[candidateLabel];
			}
		}

		oldRegionCount = regionCount;
		regionCount	= label + 1;

		for(i = 0; i < height * width; i++) segmentData[i] = labelList[linkedList[segmentData[i]].label];

		delete [] modesList;
		delete [] pointCountList;
		delete [] labelList;
		counter++;
	}
	while((counter != 1) && ((oldRegionCount - regionCount <= 0) && (counter < 11)));

	double *modesList;
	int	*pointCountList;
	int	*labelList;

	modesList = new double[bands * regionCount];
	pointCountList = new int[regionCount];
	labelList = new int [regionCount];
	
	int candidate, candidateLabel, neighCandidateLabel, thisModeCount, minRegionCount;
	double minSqDistance, neighbourDistance;
	LLinkedList *neighbour;
	
	do
	{
		minRegionCount	= 0;		

		for(i = 0; i < regionCount; i++) linkedList[i].label = i, linkedList[i].next = NULL;
		for(i = 0; i < 10 * regionCount - 1; i++) pool[i].next = &pool[i + 1];
		pool[10 * regionCount - 1].next = NULL;

		LLinkedList *freelinkedList	= pool, *node1, *node2, *oldFreeList;

		segmentData = segmentImage.GetData();
		for(i = 0; i < height; i++) for(j = 0; j < width; j++)
		{
			int curLabel = segmentData[i * width + j];

			if(j != width - 1)
			{
				int rightLabel = segmentData[i * width + j + 1];
				if(curLabel != rightLabel)
				{
					node1 = freelinkedList;
					node2 = freelinkedList->next;
					oldFreeList = freelinkedList;
					freelinkedList = freelinkedList->next->next;
					node1->label = curLabel;
					node2->label = rightLabel;

					linkedList[curLabel].Insert(node2);
					if(linkedList[rightLabel].Insert(node1)) freelinkedList = oldFreeList;
				}
			}
			if(i != height - 1)
			{
				int bottomLabel	= segmentData[(i + 1) * width + j];
				if(curLabel != bottomLabel)
				{
					node1	= freelinkedList;
					node2	= freelinkedList->next;
					oldFreeList = freelinkedList;
					freelinkedList = freelinkedList->next->next;
					node1->label = curLabel;
					node2->label = bottomLabel;

					linkedList[curLabel].Insert(node2);
					if(linkedList[bottomLabel].Insert(node1)) freelinkedList = oldFreeList;
				}
			}
		}
		
		for(i = 0; i < regionCount; i++)
		{
			if(modePointCounts[i] < minRegion)
			{
				minRegionCount++;
				neighbour = linkedList[i].next;
				
				candidate = neighbour->label;

				minSqDistance = 0;
				for(j = 0; j < bands; j++)
				{
					double el = (modes[i * bands + j] - modes[candidate * bands + j]) / sigmaLuv;
					minSqDistance += el * el;
				}
				
				neighbour = neighbour->next;
				while(neighbour)
				{
					neighbourDistance = 0;
					for(j = 0; j < bands; j++)
					{
						double el = (modes[i * bands + j] - modes[neighbour->label * bands + j]) / sigmaLuv;
						neighbourDistance += el * el;
					}
					if(neighbourDistance < minSqDistance)
					{
						minSqDistance = neighbourDistance;
						candidate = neighbour->label;
					}
					neighbour = neighbour->next;
				}
				candidateLabel = i;
				while(linkedList[candidateLabel].label != candidateLabel) candidateLabel = linkedList[candidateLabel].label;

				neighCandidateLabel = candidate;
				while(linkedList[neighCandidateLabel].label != neighCandidateLabel) neighCandidateLabel = linkedList[neighCandidateLabel].label;

				if(candidateLabel < neighCandidateLabel) linkedList[neighCandidateLabel].label = candidateLabel;
				else
				{
					linkedList[linkedList[candidateLabel].label].label = neighCandidateLabel;
					linkedList[candidateLabel].label = neighCandidateLabel;
				}
			}
		}
		for(i = 0; i < regionCount; i++)
		{
			candidateLabel = i;
			while(linkedList[candidateLabel].label != candidateLabel) candidateLabel = linkedList[candidateLabel].label;
			linkedList[i].label	= candidateLabel;
		}
		for(i = 0; i < regionCount; i++) pointCountList[i]	= 0;
		for(i = 0; i < bands * regionCount; i++) modesList[i] = 0;
		
		for(i = 0; i < regionCount; i++)
		{
			candidateLabel = linkedList[i].label;
			thisModeCount = modePointCounts[i];
			for(k = 0; k < bands; k++) modesList[bands * candidateLabel + k] += thisModeCount*modes[bands * i + k];
			pointCountList[candidateLabel] += thisModeCount;
		}
		for(i = 0; i < regionCount; i++) labelList[i] = -1;
		
		label = -1;
		for(i = 0; i < regionCount; i++)
		{
			candidateLabel = linkedList[i].label;
			if(labelList[candidateLabel] < 0)
			{
				label++;
				labelList[candidateLabel] = label;
				
				thisModeCount = pointCountList[candidateLabel];
				for(k = 0; k < bands; k++) modes[bands * label + k] = modesList[bands * candidateLabel + k] / thisModeCount;
				modePointCounts[label]	= pointCountList[candidateLabel];
			}
		}
		
		oldRegionCount = regionCount;
		regionCount = label + 1;
		
		for(i = 0; i < height * width; i++) segmentData[i] = labelList[linkedList[segmentData[i]].label];
	}
	while(minRegionCount > 0);

	delete [] modesList;
	delete [] pointCountList;
	delete [] labelList;
	delete[] linkedList;
	delete[] pool;

	delete[] modes;
	delete[] modePointCounts;
}

LKMeansSegmentation2D::LKMeansSegmentation2D(double setXyLuvRatio, int setDistance, int setIterations, int setMaxDiff, const char *setFolder, const char *setExtension) : LSegmentation2D(setFolder, setExtension)
{
	xyLuvRatio = setXyLuvRatio, distance = setDistance, iterations = setIterations, maxDiff = setMaxDiff;
}

void LKMeansSegmentation2D::Segment(LImage<double> &image, LSegmentImage &segmentImage)
{
	int width = image.GetWidth(), height = image.GetHeight();

	segmentImage.SetResolution(width, height);

	int countx = (width + distance / 2) / distance, county = (height + distance / 2) / distance, count = countx * county;
	int bands = image.GetBands(), i, j, k, l, m, n;
	
	double *means = new double[count * (2 + bands)];

	int offsetx = (width - (countx - 1) * distance) / 2, offsety = (height - (county - 1) * distance) / 2;
	for(j = 0; j < county; j++) for(i = 0; i < countx; i++)
	{
		means[((j * countx) + i) * (2 + bands)] = i * distance + offsetx;
		means[((j * countx) + i) * (2 + bands) + 1] = j * distance + offsety;
		for(k = 0; k < bands; k++) means[((j * countx) + i) * (2 + bands) + 2 + k] = image((int)means[((j * countx) + i) * (2 + bands)], (int)means[((j * countx) + i) * (2 + bands) + 1], k);
	}

	int *segCounts = new int[count];
	double *sumLuv = new double[count * (bands + 2)];

	for(l = 0; l < iterations; l++)
	{
		double *luvData = image.GetData();
		int *segData = segmentImage.GetData();

		for(j = 0; j < height; j++) for(i = 0; i < width; i++, luvData += bands, segData++)
		{
			int x = (i - offsetx + distance / 2) / distance;
			int y = (j - offsety + distance / 2) / distance;

			int minx = (x - maxDiff < 0) ? 0 : x - maxDiff;
			int maxx = (x + maxDiff > countx - 1) ? countx - 1 : x + maxDiff;
			int miny = (y - maxDiff < 0) ? 0 : y - maxDiff;
			int maxy = (y + maxDiff > county - 1) ? county - 1 : y + maxDiff;

			if(minx >= countx - 1) minx = countx - 1;
			if(maxx < 0) maxx = 0;
			if(miny >= county - 1) miny = county - 1;
			if(maxy < 0) maxy = 0;

			int bestIndex = miny * countx + minx;
			double *mean = means + bestIndex * (bands + 2);
			double bestDist = ((i - mean[0]) * (i - mean[0]) + (j - mean[1]) * (j - mean[1])) * xyLuvRatio * xyLuvRatio;
			for(k = 0; k < bands; k++) bestDist += (mean[k + 2] - luvData[k]) * (mean[k + 2] - luvData[k]);

			for(n = miny; n <= maxy; n++) for(m = (n == miny) ? minx + 1 : minx; m <= maxx; m++)
			{
				int index = n * countx + m;
				mean = means + index * (bands + 2);
				double dist = ((i - mean[0]) * (i - mean[0]) + (j - mean[1]) * (j - mean[1])) * xyLuvRatio * xyLuvRatio;
				for(k = 0; k < bands; k++) dist += (mean[k + 2] - luvData[k]) * (mean[k + 2] - luvData[k]);
				if(dist < bestDist) bestIndex = index, bestDist = dist;
			}
			*segData = bestIndex;
		}

		if(l != iterations - 1)
		{
			memset(segCounts, 0, count * sizeof(int));
			memset(sumLuv, 0, count * (bands + 2) * sizeof(double));

			luvData = image.GetData();
			segData = segmentImage.GetData();

			for(j = 0; j < height; j++) for(i = 0; i < width; i++, luvData += bands, segData++)
			{
				sumLuv[*segData * (bands + 2)] += i;
				sumLuv[*segData * (bands + 2) + 1] += j;
				for(k = 0; k < bands; k++) sumLuv[*segData * (bands + 2) + k + 2] += luvData[k];
				segCounts[*segData]++;
			}
			for(i = 0; i < count; i++) for(k = 0; k < bands + 2; k++) means[i * (bands + 2) + k] = (segCounts[i] > 0) ?  sumLuv[i * (bands + 2) + k] / (double) segCounts[i] : 0;
		}
	}
	if(segCounts != NULL) delete[] segCounts;
	if(sumLuv != NULL) delete[] sumLuv;
	if(means != NULL) delete[] means;
}

LDummySegmentation2D::LDummySegmentation2D(const char *setFolder, const char *setExtension) : LSegmentation2D(setFolder, setExtension)
{
}

void LDummySegmentation2D::Segment(LImage<double> &image, LSegmentImage &segmentImage)
{
	segmentImage.SetResolution(0, 0);
}
