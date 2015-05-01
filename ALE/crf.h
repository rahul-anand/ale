#ifndef __crf
#define __crf

#include "dataset.h"
#include "learning.h"
#include "graph.h"

class LCrfDomain;
class LCrf;

class LCrfLayer
{
	protected :
		int classNo;
		LDataset *dataset;
	public :
		LCrf *crf;
		LCrfDomain *domain;
		LCrfLayer *parent;
		unsigned char *labels, *active;
		int nodeOffset;
		int range;

		LCrfLayer(LCrf *setCrf, LCrfDomain *setDomain, LDataset *setDataset, LCrfLayer *setParent, int setRange);
		~LCrfLayer();
		virtual void Initialize(char *imageFileName, int onFly = 0) {};
		virtual void SetLabels() {};
		virtual void UnInitialize() {};
		virtual int GetNodeCount() = 0;
		virtual int GetEdgeCount() = 0;
		virtual int GetPairwiseNodeCount();
		virtual void BuildGraph(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes) {};
		virtual int UpdateLabels(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		virtual int BinaryNodes();
};

class LBaseCrfLayer : public LCrfLayer
{
	protected :
		int width, height;
	public :
		LBaseCrfLayer(LCrf *setCrf, LCrfDomain *setDomain, LDataset *setDataset, int setRange);
		~LBaseCrfLayer();

		void Initialize(char *imageFileName, int onFly = 0);
		void UnInitialize();
		int GetNodeCount();
		int GetEdgeCount();
		void BuildGraph(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		int UpdateLabels(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		int BinaryNodes();
		int GetPairwiseNodeCount();
};

class LPnCrfLayer : public LCrfLayer
{
	protected :
	public :
		int *segmentCounts, **segmentIndexes, segmentCount;
		int *baseSegmentCounts, **baseSegmentIndexes;
		LSegmentation2D *segmentation;
		double **weights, *weightSums, truncation;
		const char *segFolder, *segExtension;

		LPnCrfLayer(LCrf *setCrf, LCrfDomain *setDomain, LDataset *setDataset, LCrfLayer *setParent, LSegmentation2D *setSegmentation, double setTruncation);
		LPnCrfLayer(LCrf *setCrf, LCrfDomain *setDomain, LDataset *setDataset, LCrfLayer *setParent, const char *setSegFolder, const char *setSegExtension, double setTruncation);
		~LPnCrfLayer();

		void Initialize(char *imageFileName, int onFly = 0);
		void SetLabels();
		void UnInitialize();
		int GetNodeCount();
		int GetEdgeCount();
		void BuildGraph(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		int UpdateLabels(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		int BinaryNodes();
		int GetPairwiseNodeCount();
};

class LPreferenceCrfLayer : public LCrfLayer
{
	protected :
	public :
		LPreferenceCrfLayer(LCrf *setCrf, LCrfDomain *setDomain, LDataset *setDataset, LCrfLayer *setParent);
		~LPreferenceCrfLayer();
		int GetNodeCount();
		int GetEdgeCount();
		void BuildGraph(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		void Initialize(char *imageFileName, int onFly = 0);
		void UnInitialize();
};

class LCrfDomain
{
	protected :
	public :
		LCrf *crf;
		LDataset *dataset;
		LBaseCrfLayer *baseLayer;
		int classNo;
		const char *testFolder;

		LCrfDomain(LCrf *setCrf, LDataset *setDataset, int setClassNo, const char *setTestFolder, void (LDataset::*setRgbToLabel)(unsigned char *, unsigned char *), void (LDataset::*setLabelToRgb)(unsigned char *, unsigned char *));
		void (LDataset::*rgbToLabel)(unsigned char *, unsigned char *);
		void (LDataset::*labelToRgb)(unsigned char *, unsigned char *);
};

class LCrf
{
	private :
		LDataset *dataset;
		Graph<double, double, double> *g;
		Graph<double, double, double>::node_id *nodes;

	public :
		LList<LCrfDomain *> domains;
		LList<LCrfLayer *> layers;
		LList<LPotential *> potentials;
		LList<LFeature *> features;
		LList<LSegmentation2D *> segmentations;
		LList<LLearning *> learnings;

		LCrf(LDataset *setDataset);
		~LCrf();

		void Segment(char *imageFileName);
		void Segment(LList<char *> &imageFiles, int from = 0, int to = -1);
		void TrainFeatures(LList<char *> &imageFiles);
		void EvaluateFeatures(LList<char *> &imageFiles, int from = 0, int to = -1);
		void TrainPotentials(LList<char *> &imageFiles);
		void EvaluatePotentials(LList<char *> &imageFiles, int from = 0, int to = -1);
		void Solve(LList<char *> &imageFiles, int from = 0, int to = -1);

		void InitSolver(char *imageFileName, LLabelImage &labelImage);
		void UnInitSolver();
		void Solve(char *imageFileName);

		int Expand(LCrfDomain *domain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		void Confusion(LList<char *> &imageFiles, char *confusionFileName);
		void Confusion(LList<char *> &imageFiles, char *confusionFileName, int maxError);
};

#endif