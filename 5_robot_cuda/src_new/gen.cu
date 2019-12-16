#include <stdio.h>
#include "type.cuh"
#include "select.cuh"
#include "init.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//double EvalPopulation(IPTR pj, int start, int end, Population *p);
void   Halve(IPTR o, IPTR n, Population *p);
int Generation0(IPTR oldpop, IPTR newpop, int t, Population *p, Functions *f) 
{
	printf("got here -1.01\n");
  int i, p1, p2;

  IPTR pj, pjplus1, om1, om2; 

  for(i = 0; i < p->popsize; i += 2){

    p1 = f->FindMate(oldpop, p->popsize, p);
    p2 = f->FindMate(oldpop, p->popsize, p);

    pj = &(newpop[i]);
    pjplus1 = &(newpop[i+1]);
    om1 = &(oldpop[p1]);
    om2 = &(oldpop[p2]);

    f->Crossover(om1, om2, pj, pjplus1, p);

    pj->parent1 = p1;
    pj->parent2 = p2;

    pjplus1->parent1 = p2;
    pjplus1->parent2 = p1;

  }

  //calculate the number of blocks needed for popsize
  int blockSize = 256;
  int numBlocks = (p->popsize + blockSize - 1) / blockSize;

  //int fitnesses[2];



  INDIVIDUAL* shared_pop;
  cudaMallocManaged(&shared_pop, p->popsize * sizeof(INDIVIDUAL));

  int* shared_chroms;
  cudaMallocManaged(&shared_chroms, p->popsize*p->chromLength*sizeof(int));

  //printf("got here -2.1 \n");
  for (int i = 0; i < p->popsize; i++)
  {
	  shared_pop[i] = p->newpop[i];
	  pj = &shared_pop[i];
	  for (int j = 0; j < p->chromLength; j++)
	  {
		  shared_chroms[i*pj->chromLen + j] = pj->chrom[j];
	  }
  }

  //EvalPopulation <<<numBlocks, blockSize >>>(newpop, 0, p->popsize, p, fitnesses);
  EvalPopulation << <numBlocks, blockSize >> >(shared_pop, 0, p->popsize, shared_chroms);

  //printf("got here -1.1: popsize = %d newpop = \n", p->popsize);
  
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  IPTR op;
  for (int i = 0; i < p->popsize; i++)
  {
	  pj = &p->newpop[i];
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


  return 0;

}


int CHC(IPTR oldpop, IPTR newpop, int t, Population *p, Functions *f)
{
  int i, p1, p2;
  IPTR pj, pjplus1, om1, om2;

  for(i = p->popsize; i < p->lambda*p->popsize; i += 2){

    p1 = f->FindMate(oldpop, p->popsize, p);
    p2 = f->FindMate(oldpop, p->popsize, p);

    pj = &(oldpop[i]);
    pjplus1 = &(oldpop[i+1]);
    om1 = &(oldpop[p1]);
    om2 = &(oldpop[p2]);

    /*********
    fprintf(stdout, "CHC: i:%d; p1: %d; p2: %d\n", i, p1, p2);
    fprintf(stdout, "CHC: ol1: %d; ol2: %d\n", om1->chromLen, om2->chromLen);
    fflush(stdout);
    *******************/    
    f->Crossover(om1, om2, pj, pjplus1, p);

    pj->parent1 = p1;
    pj->parent2 = p2;

    pjplus1->parent1 = p2;
    pjplus1->parent2 = p1;

    
  }
  /******************
  fprintf(stdout, "Before evaluating population \n");
  ******************/

  //calculate the number of blocks needed for popsize
  int blockSize = 256;
  int numBlocks = (p->popsize + blockSize - 1) / blockSize;

  
  INDIVIDUAL* shared_pop;
  cudaMallocManaged(&shared_pop, p->popsize * p->lambda * sizeof(INDIVIDUAL));

  int* shared_chroms;
  cudaMallocManaged(&shared_chroms, p->popsize*p->lambda*p->chromLength*sizeof(int));

  //printf("got here -2.1 \n");
  for (int i = 0; i < p->popsize * p->lambda; i++)
  {
	  shared_pop[i] = p->oldpop[i];
	  pj = &shared_pop[i];
	  for (int j = 0; j < p->chromLength; j++)
	  {
		  shared_chroms[i*pj->chromLen + j] = pj->chrom[j];
	  }
  }


  //EvalPopulation << <numBlocks, blockSize >> > (oldpop, p->popsize, p->popsize * p->lambda, p, fitnesses);
  EvalPopulation << <numBlocks, blockSize >> > (shared_pop, p->popsize, p->popsize * p->lambda, shared_chroms);

  //printf("fitnesses[0] = %d\n", fitnesses[0]);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  IPTR op;
  for (int i = p->popsize; i < p->popsize * p->lambda; i++)
  {
	  pj = &p->oldpop[i];
	  op = &shared_pop[i];
	  pj->objfunc = op->objfunc;
	  pj->fitness = p->maxConst - pj->objfunc;
  }

  cudaFree(shared_pop);
  cudaFree(shared_chroms);

  Halve(oldpop, newpop, p);

  return 0;

}

