#ifndef __TYPE_H__

#define __TYPE_H__

#include "app.cuh"

typedef struct {
  ChromType *chrom;
  ChromTypeB *backup;
  
  int chromLen;
  double x;

  double fitness, scaledFitness;
  double objfunc;

  int parent1, parent2;
  double dx, dy, cx, cy;

  Appstruct app; /* from app.h */

} INDIVIDUAL;

typedef INDIVIDUAL *IPTR;
typedef INDIVIDUAL individual;

typedef struct {
  IPTR oldpop;
  IPTR newpop;

  int    verbose;
  double seed;
  int    maximize;
  double maxConst;

  int generation;
  int maxgen;
  int popsize;
  int chromLength;
  double max, min, avg;
  int    maxi, mini, bigMaxi;
  double bigMax;
  int    bigGen, bigInd;
  
  double sumFitness, scaledSumFitness;
  double smax, smin;
  double scaleConstA, scaleConstB;
  
  float pCross, pMut;
  int   xType;
  int   nXPoints;

  float scaleFactor;
  int   lambda;
  
  float injectFraction; /* if > 0 inject cases from casebase else not*/
  int   injectPeriod;
  int   injectStop;
  int   saveCases;      /* Should I save cases to casebase */
  int   dMetric;
  char *caseFileName;
  char *nCFile; /* contains number of cases. 
		   Note should be folded into caseFileName */
  int  nCases;
  int  nCurrentCases;

  char *iFile, *oFile, *pidFile, *fitFile;
  char *phenoFile, *errorFile, **gFile;

  char *appInfile;

} Population;

struct tfuncs{

  /* selectors, crossers, mutators */
  int (* FindMate)(IPTR curPop, int size, Population *p);
  void (* Crossover) (IPTR p1, IPTR p2, IPTR c1, IPTR c2, Population *p); 
  int (* CurrentGA)(IPTR op, IPTR np, int t, Population *p, struct tfuncs *f);
  ChromType (* Mutate)(IPTR c, int i, Population *p); 

  /* injectors, similarity measurerers */

  int (* GetIndexIndividual)(IPTR pop, int popsize);
  void (* ApplyMetric)(double *sort, int *rank, int popsize, int howmany);
  
  double (* DistanceMetric)(ChromType *p1, ChromType *p2, 
			    int size1, int size2, ChromType **tmplate);
};

typedef struct tfuncs Functions;

#endif

