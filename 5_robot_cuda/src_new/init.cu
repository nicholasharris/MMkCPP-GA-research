#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <unistd.h>

#include "defaults.cuh"
#include "type.cuh"

#include "init.cuh"
#include "utils.cuh"
#include "case.cuh"
#include "finders.cuh"
#include "random.cuh"
#include "dist.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//double EvalPopulation(IPTR pj, int start, int end, Population *p);

void GetInputParameters(Population *p, Functions *f)
{ /* initialize Population Params */

  FILE *inpfl;
  int tmpint;
  char ofile[1024];
  /* temporary place holders for filenames */
  
  if( (inpfl = fopen(p->iFile, "r")) == NULL){ 
    printf("error in opening file %s \n", p->iFile);
    exit(1);
  }
  
  skipline(inpfl);
  
  printf(" Enter population size - popsize-> "); 
  fscanf(inpfl,"%d",&(p->popsize));
  printf("popsize = %d\n", p->popsize);
  skipline(inpfl);
  
  printf(" Enter chromosome length - lchrom-> "); 
  fscanf(inpfl,"%d",&(p->chromLength));
  printf("lchrom = %d\n", p->chromLength);
  skipline(inpfl);
  
  printf(" Enter max. generations - maxgen-> "); 
  fscanf(inpfl,"%d",&(p->maxgen));
  printf("maxgen = %d\n", p->maxgen);
  skipline(inpfl);
  
  printf(" Enter kind of crossover (0(npoint),1(PMX),2(OX), 3(Greedy),..)-> "); /* Assume binary Xover*/
  fscanf(inpfl,"%d", &(p->xType)); /* pmx, mx, 2point, etc */
  switch (p->xType) {
  case 0: 
    f->Crossover = NPointCrossover;
    break;
  case 1:
    f->Crossover = PMX;
    printf(" Xover: PMX for sequential representations\n");
    break;
  case 2:
    f->Crossover = OX;
    printf(" Xover: OX for sequential representations\n");
    
    break;
  case 3:
    f->Crossover = Greedy;
    printf(" Xover: Greedy for sequential representations\n");
    
    break;
  default:
    printf(" Default Xover: Simple 2 point crossover \n");
    f->Crossover = NPointCrossover;
    p->nXPoints = 2;
    break;
  }
  skipline (inpfl);
  
  printf(" Enter number of crossover points (used only for npoint xover)->\n");
  fscanf(inpfl, "%d", &(p->nXPoints));
  if(p->xType == 0)
    printf(" Xover: Simple N point where n is %d \n", p->nXPoints);
  
  skipline(inpfl);
  

  printf(" Enter crossover prob -> "); 
  fscanf(inpfl,"%f",&(p->pCross));
  printf("Crossover probability = %f\n", p->pCross);
  skipline(inpfl);

  printf(" Enter mutation prob -> "); 
  fscanf(inpfl,"%f",&(p->pMut));
  printf("Mutation probability = %f\n", p->pMut);
  skipline(inpfl);

  printf(" Enter file name for graph output -fname-> ");
  fscanf(inpfl,"%s", ofile);
  free(p->oFile);
  p->oFile = NULL;
  strcreate(&(p->oFile), ofile);
  printf(" Save file is %s\n", p->oFile);
  skipline(inpfl);
    
  printf(" Enter GAflavor (0/1) -> ");
  fscanf(inpfl,"%d", &(tmpint));
  switch (tmpint) {
  case 0: 
    printf(" GAFlavor: Offspring replace parents \n");
    f->CurrentGA = Generation0;
    break;
  case 1:
    f->CurrentGA = CHC;

    break;
  default:
    printf(" Default: Canonical GA: Offspring replace parents \n");
    f->CurrentGA = Generation0;
  }
  skipline(inpfl);

  printf(" Enter Lambda (used only if CHC) ->\n");
  fscanf(inpfl, "%d", &(p->lambda));
  if(tmpint == 1)
    printf(" GAFlavor: CHC with lambda %d \n", p->lambda);
  skipline(inpfl);

  printf(" Enter MateFinder choice -> ");
  fscanf(inpfl,"%d", &(tmpint));
  switch (tmpint) {
  case 0: 
    printf(" Fitness Proportional \n");
    f->FindMate = Roulette;
    break;
  case 1:
    printf(" Random \n");
    f->FindMate = RandomMate;
    break;
  default:
    printf(" Canonical: Fitness Proportional \n");
    f->FindMate = Roulette;
    break;
  }
  skipline(inpfl);

  printf(" Enter Scaling Factor -> ");
  fscanf(inpfl,"%f", &(p->scaleFactor));
  printf(" Scale Factor %f\n", p->scaleFactor);
  skipline(inpfl);

  printf(" Enter converter to maximizing problem:\n A large constant");
  printf(" from which to subtract the obj. function value.\n");
  printf(" Used only if this is a minimization problem \n");
  fscanf(inpfl,"%lf", &(p->maxConst));
  printf(" maxConst %f\n", p->maxConst);
  fprintf(stderr, " maxConst %f\n", p->maxConst);
  skipline(inpfl);
    
  printf(" Should I save cases? ");
  fscanf(inpfl,"%d", &(p->saveCases));
  if(p->saveCases){
    printf(" Saving cases \n");
  } else {
    printf(" Not Saving cases \n");
  }

  printf(" If I am going to use the case-base enter Injection Fraction ");
  fscanf(inpfl,"%f", &(p->injectFraction));
  printf(" Injection Fraction is %f\n", p->injectFraction);

  if(p->injectFraction <= 0.0) {
    printf(" I am NOT going to use the case-base\n "); 
  }
  
  
  printf(" Enter Inject Period ");
  fscanf(inpfl,"%d", &(p->injectPeriod));
  printf(" Injection Period is %d\n", p->injectPeriod);
  
  printf(" Enter Inject Stop time ");
  fscanf(inpfl,"%d", &(p->injectStop));
  printf(" Injection stop time is %d\n", p->injectStop);
  
  printf(" Injection strategy -> ");
  fscanf(inpfl, "%d", &tmpint);
  switch (tmpint) {
  case 0: 
    printf(" Using Closest to Best \n ");
    f->GetIndexIndividual = GetBest; 
    f->ApplyMetric = FindNClosest; 
    break;
  case 1:
    printf(" Using Probabilistic closest to best \n ");
    f->GetIndexIndividual = GetBest;
    f->ApplyMetric = FindNCloseProbable;
    break;
  case 2:
    printf(" Using Furthest from Worst  \n");
    f->GetIndexIndividual = GetWorst;
    f->ApplyMetric = FindNFurthest;
  case 3:
    printf(" Using Probabilistic Furthest from Worst \n ");
    f->GetIndexIndividual = GetWorst;
    f->ApplyMetric = FindNFurthProbable;
    
  default:
    printf(" Using Closest to Best \n ");
    f->GetIndexIndividual = GetBest;
    f->ApplyMetric = FindNClosest;
    break;
  }
  
  printf(" Distance Metric -> ");
  fscanf(inpfl, "%d", &(p->dMetric));
  switch (p->dMetric) {
  case 0: 
    f->DistanceMetric = HamDist;
    printf(" Distance Metric: Hamming Distance \n");
    break;
  case 1:
    f->DistanceMetric = Euclidean;
    printf(" Distance Metric: Euclidean Distance \n");
    break;
  case 2:
    f->DistanceMetric = LCS;
    printf(" Distance Metric: Longest Common Substring \n");
    break;
  default:
    f->DistanceMetric = HamDist;
    printf(" Distance Metric: Hamming Distance \n");
    break;
  }
  skipline(inpfl);

  fclose(inpfl);
  printf("\n");
}

void PopulationAllocate(Population *p)
{
  IPTR pj;
  int i;
  p->newpop = (IPTR) malloc(sizeof(INDIVIDUAL) * p->popsize * p->lambda);
  p->oldpop = (IPTR) malloc(sizeof(INDIVIDUAL) * p->popsize * p->lambda);
  for(i = 0; i < p->popsize * p->lambda; i++){
    pj = &(p->newpop[i]);
    pj->chrom = (ChromType *) malloc (sizeof(ChromType) * p->chromLength);
    pj->backup = (ChromType *) malloc (sizeof(ChromType) * p->chromLength);

    pj = &(p->oldpop[i]);
    pj->chrom = (ChromType *) malloc (sizeof(ChromType) * p->chromLength);
    pj->backup = (ChromType *) malloc (sizeof(ChromType) * p->chromLength);

    pj->chromLen = p->chromLength;
  }
}


void Initpop(Population *p)
{ /* initialize a random population */

  int i, j;
  IPTR pj, op, np;

  fprintf(stdout, "Lambda = %d\n", p->lambda);

  PopulationAllocate(p);

  op = p->oldpop;
  np = p->newpop;
  
  for (i = 0; i < p->popsize; i++){
    pj = &(op[i]);
    pj->chromLen = p->chromLength;
    for (j = 0; j < pj->chromLen; j++){
      pj->chrom[j] = Flip(0.5); /* assume default initialization to int*/
    }
    AppInitChrom(pj);
    pj->parent1 = pj->parent2 = 0;
  }
  
  //calculate the number of blocks needed for popsize
  int blockSize = 256;
  int numBlocks = (p->popsize + blockSize - 1) / blockSize;

  //printf("got here -2 \n");

  INDIVIDUAL* shared_pop;
  cudaMallocManaged(&shared_pop, p->popsize * sizeof(INDIVIDUAL));

  int* shared_chroms;
  cudaMallocManaged(&shared_chroms, p->popsize*p->chromLength*sizeof(int));

  //printf("got here -2.1 \n");
  for (int i = 0; i < p->popsize; i++)
  {
	  shared_pop[i] = p->oldpop[i];
	  pj = &shared_pop[i];
	  for (int j = 0; j < p->chromLength; j++)
	  {
		  shared_chroms[i*pj->chromLen + j] = pj->chrom[j];
	  }
	  
  }

  //printf("got here -2.1 \n");
 // pj = &shared_pop[0];

  //printf("chrom1: %d\n", pj->chrom[0]);

 // pj = &p->oldpop[0];

  //printf("chrom1: %d\n", pj->chrom[0]);
  
  //printf("got here -2.3 \n");
  //EvalPopulation << <numBlocks, blockSize >> > (p->oldpop, 0, p->popsize, p, fitnesses);
  EvalPopulation << <numBlocks, blockSize >> > (shared_pop, 0, p->popsize, shared_chroms);


  /*
  for (int i = 0; i < p->popsize; i++)
  {
	  p->oldpop[i]->fitness = shared_pop[i]->fitness;
  }*/

  //printf("fitnesses[0] = %d\n", fitnesses[0]);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  for (int i = 0; i < p->popsize; i++)
  {
	  pj = &p->oldpop[i];
	  op = &shared_pop[i];
	  pj->objfunc = op->objfunc;
	  pj->fitness = p->maxConst - pj->objfunc;
  }

  /*
  pj = &p->oldpop[0];
  int result = pj->fitness;
  printf("fitness = %d\n", result);*/

  cudaFree(shared_pop);
  cudaFree(shared_chroms);

  //printf("got here -2.4\n");
}

void Initreport(Population *p)
{
  /*  FILE *fp; */

  printf("Starting GA...\n");
  InitGooguReport(p->oldpop, p);

  Report(p->generation, p->oldpop, p);

  /**********************************************  
  if( (fp = fopen(p->caseFileName,"a")) == NULL){
    printf("error in opening file %s \n", p->caseFileName);
    exit(1);
  }else{
    SaveCase(fp, &(p->oldpop[p->maxi]), p->generation, p);
    fflush(fp);
    fclose(fp);
  }
  ************************************************/
}

void InitGooguReport(IPTR op, Population *p)
{
  FILE *fp;

  InitPhenoPrint(op, p->phenoFile, p); /* in app.c */

  if( (fp = fopen(p->fitFile, "w")) == NULL){
    printf("error in opening file fitFile %s \n", p->fitFile);
    exit(1);
  }else{
    fprintf(fp, "4 \n");/* number of columns to read */
    fprintf(fp, "gen.objFunc max avg min \n");
    fclose(fp);
  }

  /*****************************************************
  if( (fp = fopen(p->errorFile,"a")) == NULL){
    printf("error in opening file %s \n",fname);
    exit(1);
  }else{
    fprintf(fp, "2 \n");
    fprintf(fp, "gen makespan \n");
    fclose(fp);
  }
  *******************************************************/
}

void Initialize(int argc, char *argv[],
	Population *p, Functions *f)
{
	/* initialize everything */
	char ncfile[1024];
	int c;


	/* defaults are in defaults.h */
	p->seed = SEED;
	p->lambda = LAMBDA;
	p->min = p->max = p->avg = -1.0;

	/* 1st arg to strcreat shd be null */
	p->oFile = NULL;
	p->iFile = p->appInfile = NULL;
	p->phenoFile = p->errorFile = NULL;
	p->gFile = NULL;
	p->pidFile = p->fitFile = NULL;
	p->caseFileName = p->nCFile = NULL;

	strcreate(&(p->oFile), OUTPUTFILE);
	strcreate(&(p->caseFileName), CASEFILE);
	strcpy(ncfile, CASEFILE);
	strcat(ncfile, ".nmc");
	strcreate(&(p->nCFile), ncfile);

	p->popsize = POPSIZE;
	p->maxgen = MAXGEN;
	p->chromLength = CHROMLENGTH;
	p->pCross = PCROSS;
	p->pMut = PMUT;
	p->nXPoints = NXPOINTS;
	p->scaleFactor = SCALEFACTOR;
	p->maximize = TRUE;

	p->injectFraction = 0.0;
	p->injectPeriod = 0;
	p->injectStop = MAXGEN / 2;
	p->saveCases = FALSE;

	f->Crossover = CROSSOVER;
	f->CurrentGA = CURRENTGA;
	f->FindMate = FINDMATE;

	/* end defaults */

	/* input file overides command line options */
	p->seed = atof(argv[7]);
	p->iFile = argv[2];
	p->appInfile = argv[4];
	p->maximize = FALSE;
	//while ((c = getopt(argc, argv, "i:o:s:a:vM")) != EOF){

	//  switch (c) {

	//  case 'i':
	//    strcreate(&(p->iFile), optarg);
	//    break;
	//    
	//  case 'a':
	//    strcreate(&(p->appInfile), optarg);
	//    break;
	//    
	//  case 'M':
	//    p->maximize = FALSE;
	//    //      tmp = atoi(optarg);
	//    break;

	//  case 'o':
	//    free(p->oFile); p->oFile = NULL;
	//    strcreate(&(p->oFile), optarg);
	//    break;

	//  case 's':
	//    p->seed = (double) atof(optarg);
	//    break;

	//  case 'v':
	//    p->verbose = TRUE;
	//    break;

	//  default:
	//    Usage();
	//    exit(EXIT_FAILURE);
	//  }
	//}

	if (p->iFile != NULL) {
		GetInputParameters(p, f);
	}

	Randomize(p->seed);

	/* Initialize globals */

	p->nCases = 0;
	p->nCurrentCases = 0;
	if (p->injectFraction > 0.0) {
		p->nCases = FindNCases(p->nCFile);
	}
	p->bigMaxi = -1;

	p->generation = 0;

	if (p->appInfile != NULL) /* from command line */
		AppInit(p->appInfile, p);

	InitGA(p, f);

	printf("Done initializing\n");
}

void Usage()
{
  fprintf(stderr, 
	  "ga -i infile -o ofile -a application file -s seed -v verbose -M\n");
}


void InitGA(Population *p, Functions *f)
{
  InitFiles(p);

  Initpop(p);

  fprintf(stdout, "After allocating population storage\n");

  Statistics(p->oldpop, p);

  if(p->injectFraction > 0.0){
    InitLoadCases(p->caseFileName, 
		  p->oldpop, p->generation, p->injectFraction, p );
    printf("Read in Initial Case Base\n");
    LoadCases(p->oldpop, p->generation, p->injectFraction, p, f);
    printf("Loaded initial cases\n");
    Statistics(p->oldpop, p);
  }
  
  Initreport(p);

}



void InitFiles(Population *p)
{
  char phenotypeFile[1024], errorFile[1024], pidFile[1024];
  char fitFile[1024];

  /* generate filenames */
  strcpy(phenotypeFile, p->oFile);
  strcat(phenotypeFile, ".pheno"); 

  strcpy(errorFile, p->oFile);
  strcat(errorFile, ".error");

  strcpy(pidFile, p->oFile);
  strcat(pidFile, ".pid");

  strcpy(fitFile, p->oFile);
  strcat(fitFile, ".fit");

  /* create strings just big enough to hold filenames*/
  strcreate(&(p->phenoFile), phenotypeFile);
  strcreate(&(p->errorFile), errorFile);
  strcreate(&(p->pidFile), pidFile);
  strcreate(&(p->fitFile), fitFile);
  
  fprintf(stdout, "PhenoFile: %s\n", p->phenoFile);
  fprintf(stdout, "errorFile: %s\n", p->errorFile);
  fprintf(stdout, "pidFile: %s\n", p->pidFile);
  fprintf(stdout, "fitFile: %s\n", p->fitFile);
  fflush(stdout);
}
