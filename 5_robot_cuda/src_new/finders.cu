#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>  
#include "type.cuh"
#include "utils.cuh"
#include "random.cuh"

void SwapInt(int *x, int *y);
//int Proportional(double *sort, double sum, int size);
int Proportional(double *sort, double sum, int size);

void FindNFurthest(double *sort, int *rank, int size, int howmany)
{
  int i, max, current;
  
  max = 0;
  for(current = 0; current < howmany - 1 ; current++) {
    max = current;
    /* Find Next best */
    for(i = current + 1; i < size; i++) {
      if(sort[rank[i]] > sort[rank[max]]) {
	max = i;
      }
    }
    SwapInt(&rank[current], &rank[max]);
  }
}

void FindNFurthProbable(double *sort, int *rank, int size, int howmany)
{ /* Probabilistically find furthest from the worst */
  int i;
  double sum = 0.0;

  for(i = 0; i < size; i++) {
    sum += (double) sort[i];
  }
  for (i = 0; i < howmany; i++){
    rank[i] = Proportional(sort, sum, size);
  }
}

void FindNClosest(double *sort, int *rank, int size, int howmany)
{
  int i, max, current;
  
  max = 0;
  for(current = 0; current < howmany - 1  ; current++) {
    max = current;
    /* Find Next best */
    for(i = current + 1; i < size; i++) {
      if(sort[rank[i]] < sort[rank[max]]) {
	max = i;
      }
    }
    SwapInt(&rank[current], &rank[max]);
  }
}

void FindNCloseProbable(double *sort, int *rank, int size, int howmany)
{ /* Probabilistically find closest to the best */
  int i;
  double sum = 0.0;
  int max, min;

  max = sort[0];
  min = sort[0];

  for(i = 0; i < size; i++) { /* find max for the next step */
    sum += (double) sort[i];  /* find sum for Proportional selection */
    if(max < sort[i])
      max = sort[i];
    if(min > sort[i])
      min = sort[i];
    
  }
  for(i = 0; i < size; i++){  /* make max the min to find closest */
    sort[i] = min + max - sort[i]; /* need to add min */
  }                                /* to get back original values */
  for (i = 0; i < howmany; i++){
    rank[i] = Proportional(sort, sum, size);
  }
}

void FindNPopRandom(double *sort, int *rank, int size, int howmany)
{ /* injecting random individuals from the case base */
  /* Do we need to go through the pain of searching for "appropriate"
     individuals or should we simply inject any old individual in the
     case base. Remember that the case base contains "good" (in some 
     context) individuals                                          */

  Shuffle(rank, size); 

}


void FindNRanRandom(double *sort, int *rank, int size, int howmany)
{ 
  int i;
  /* Quite complicated since we have to generate random individuals*/
  /* we would like this to generate completely random individuals  */
  /* one of the tests that need to be carried out for cigar        */
  for (i = 0; i < howmany; i++){
    rank[i] = Rnd(0, size - 1);
  }
  /* this implementation is incorrect                              */
  /* A correct implementation would                                */
  /*            fill seedCases in GetCases in case.c with randomly */
  /* generated individuals                                         */
}


/************************************************************/
/* FindNWorst finds the worst in a */
/* population of individuals, NOT related to case-injection */
/* strategies                                               */
/************************************************************/

void FindNWorst(INDIVIDUAL *p, int *rank, int size, int howmany)
{ /* To choose individuals that die in population
     for replacement by CBR */
  int i, min, current;
  
  min = 0;

  for(current = 0; current < howmany - 1  ; current++) {
    min = current;
    /* Find Next best */
    for(i = current + 1; i < size ; i++) {
      if((p + rank[i])->fitness < (p + rank[min])->fitness) {
	min = i;
      }
     }
    SwapInt(&rank[current], &rank[min]);
  }
}


void FindNBest(INDIVIDUAL *p, int *rank, int size, int howmany)
{/* Used by halve to implement CHC -- not used by cbr stuff at all */
  int i, max, current;                                                 
  max = 0;
  for(current = 0; current < howmany - 1  ; current++) { 
    max = current;
    /* Find Next best */ 
    for(i = current + 1; i < size ; i++) { 
      if((p + rank[i])->fitness > (p + rank[max])->fitness) { 
        max = i; 
      } 
     } 
    SwapInt(&rank[current], &rank[max]); 
  }
}
