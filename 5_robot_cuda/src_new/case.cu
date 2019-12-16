#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "case.cuh"
#include "type.cuh"
#include "random.cuh"
#include "utils.cuh"
#include "finders.cuh"
#include "dist.cuh"



//extern "C" int GetBest(IPTR pop, int size);


IPTR caseBase;
int *sIndex, *hamDist;

void TourPrint(FILE *fp, IPTR pj, char *name);


void LoadCases(IPTR pop, int gen, float frac, Population *p, Functions *f)
{
  int nseeds, newSeeds, i, index;
  int *rank;
  IPTR seedCases;

  if(p->nCases <= 0) return;
  if(p->injectFraction   <= 0.0) return;
  nseeds = (int) ((float) p->popsize * (float) p->injectFraction);
  if (p->nCases < nseeds) {
    nseeds = p->nCases;
  }

  rank = (int *) malloc (sizeof(int) * p->popsize);
  if (rank == NULL) {
    perror("error in malloc (rank)\n"); 
  }

  seedCases = AllocateIndividuals(nseeds, p->chromLength);
  /* (IPTR) malloc (sizeof(INDIVIDUAL) * nseeds);*/
  if (seedCases == NULL) {
    perror("error in malloc (seedCases)\n"); 
  }

  index = f->GetIndexIndividual(pop, p->popsize);
  /*   printf("Before GetCases\n");*/

  newSeeds = GetCases( &pop[index], seedCases, nseeds, p, f);
  /* newSeeds may be less than nseeds */
  /*  printf("after GetCases\n");*/
#ifdef DEBUG
  PrintCases(debugFName, seedCases, nseeds, pop, best);
#endif

  for(i = 0; i < p->popsize; i++) {
    rank[i] = i;
  }
  FindNWorst(pop, rank, p->popsize, newSeeds);
  /*  printf("after FindNworst %d\n", nseeds);  */
  for(i = 0; i < newSeeds; i++){
    IndividualCopy(&seedCases[i], &pop[rank[i]]); 
    /*    pop[rank[i]].fitness = eval_org(&pop[rank[i]]); */
    /* Done in GetCases */
  }
  free (rank);
  free (seedCases);
}

int GetBest(IPTR pop, int size)
{
  int i;
  int max = 0;
  for(i = 1; i < size; i++){
    if(pop[i].fitness > pop[max].fitness){
      max = i;
    }
  }
  return max;
}

int GetWorst(IPTR pop, int size)
{
  int i;
  int min = 0;
  for(i = 1; i < size; i++){
    if(pop[i].fitness < pop[min].fitness){
      min = i;
    }
  }
  return min;
}

void SaveCase(FILE *fp, IPTR pj, int gen, Population *p)
{
  int i;

  if(p->saveCases) {
    fprintf(fp, "%5d %i ", gen, pj->chromLen);
    for(i = 0; i < pj->chromLen ; i++) {
      fprintf(fp, "%i ", pj->chrom[i]);
    }
    fprintf(fp, " %f %f\n", pj->fitness, pj->scaledFitness);
    p->nCurrentCases++;
  }
  return;
}

//temporarilyt disabled in switch to cuda?
int GetCases(IPTR pj, IPTR iCases, int howmany, Population *p, Functions *f)
{
  int i, ncopied = 0;
  int *sIndex;
  double *dist;
  
  if(p->nCases <= 0) 
    return 0;
  
  sIndex = (int *) malloc (sizeof(int) * p->nCases);
  if (sIndex == NULL) {
    perror("error in malloc (sIndex)\n"); 
  }
  dist = (double *) malloc (sizeof(double) * p->nCases);
  if (dist == NULL) {
    perror("error in malloc (dist)\n"); 
  }
  for(i = 0; i < p->nCases; i++) {
    dist[i] = 
      f->DistanceMetric(pj->chrom, (&(caseBase[i]))->chrom, 
			pj->chromLen, (&(caseBase[i]))->chromLen,
			&(caseBase[i].backup));
    /* This is the address of backup so I can change what backup points to
       Needed in sequential representations to deal with varying size 
       cases.
     */
  }
  for(i = 0; i < p->nCases; i++) {
    sIndex[i] = i;
  }
  f->ApplyMetric(dist, sIndex, p->nCases, howmany);

  ncopied = 0;
  for(i = 0; 
      (i < p->nCases) && (ncopied < howmany) && (ncopied < p->nCases); 
      i++) { 
    if(dist[sIndex[i]] != 0.0) { 
      IndividualCopy(&caseBase[sIndex[i]], &iCases[ncopied]);
      if(p->xType > 0 && p->dMetric == LCSD) {
	FixCopiedIndividual(&iCases[ncopied], p);
	//	TourPrint(stdout, &iCases[ncopied], "GetCases");
      }/* to copy backup to chrom before evaluation*/

	  iCases[ncopied].fitness = 1.0; //Eval(&(iCases[ncopied]));  //disabled in switch to cuda
		  printf("CASES Was reached\n");
      /* how do I parallelize this? */
      iCases[ncopied].dx = dist[sIndex[i]];
      ncopied++;
    }
  }
  /*************************************************************  
  ncopied = 0;
  for(i = 0; i < howmany; i++) {
    IndividualCopy(&caseBase[sIndex[i]], &iCases[i]);
    iCases[i].fitness = Eval(&(iCases[i]));
    iCases[i].dx = dist[sIndex[i]];
    ncopied++;
  }
  *************************************************************/
  
  free (sIndex);
  free (dist);
  return ncopied;
}


void ReadCase(FILE *fp, IPTR pj, Population *p)
{
  int t, i, len;

  fscanf(fp, "%i %i", &t, &len);
  if(p->xType < 1) {
    for(i = 0; i < len; i++) {
      fscanf(fp, "%i", &(pj->chrom[i]));
      // All I/O should be abstracted out
    }
    pj->chromLen = p->chromLength;
  } else { //sequential (TSP) representation
    GetSetSeqChrom(fp, pj, p, len);
    //    TourPrint(stdout, pj, "ReadCase");
  }
  fscanf(fp, "%lf %lf", &(pj->fitness), &(pj->scaledFitness));

}

void GetSetSeqChrom(FILE *fp, IPTR pj, Population *p, int len)
{/* This frees up the old chromosome, and replaces it with 
    allocated space for a chromosome of length len, then reads
    new chromosome from file* fp*/
  int i;
  
  free(pj->chrom);
  free(pj->backup);
  pj->chrom  = (ChromType *) calloc ((size_t) len, sizeof(ChromType));
  pj->backup = (ChromType *) calloc ((size_t) len, sizeof(ChromType));
  
  pj->chromLen = len;
  
  for(i = 0; i < len; i++){
    fscanf(fp, "%i", &(pj->chrom[i])); 
    // All I/O should be abstracted out
  }
}

int FindNCases(char *ncfile)
{
  FILE *fp;
  int tmp;

  if((fp = fopen(ncfile, "r")) == NULL){
    fprintf(stdout, "no cases in case base\n");
    return 0;
  } else {
    fscanf(fp, "%d", &tmp);
    fclose(fp);
    return tmp;
  }
}

void StoreNcases(char *ncfile, int ncases, int nCurrentCases)
{
  FILE *fp;

  if((fp = fopen(ncfile, "w")) == NULL){
    fprintf(stdout, "problem in opening %s \n", ncfile);
    exit(1);
  } else {
    fprintf(fp, "%d\n", ncases + nCurrentCases);
    fclose(fp);
  }
}

void InitLoadCases(char *caseFile, IPTR pop, int gen, int perc, Population *p)
{
  FILE *fp;
  int i;
  p->nCases = FindNCases(p->nCFile);
  if (p->nCases <= 0) return;
  caseBase = AllocateIndividuals(p->nCases, p->chromLength);
  if(caseBase == NULL){
    perror("Malloc failure for caseBase\n");
    exit(1);
  }
  if ((fp = fopen(caseFile, "r")) == NULL) {
    fprintf(stderr, "InitLoadCases: Cannot open %s for reading\n", caseFile);
    exit(1);
  } 
  for(i = 0; i < p->nCases; i++) {
    ReadCase(fp, &caseBase[i], p);
  }
  fclose(fp);
}
