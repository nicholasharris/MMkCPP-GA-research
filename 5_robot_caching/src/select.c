#include "type.h"
#include "random.h"

/* for cigar */
int Proportional(double *sort, double sum, int size, Population *p)
{
  double rand, partsum;
  int j;

  partsum = 0.0;
  rand = FRandom() * sum;

  j = -1;

  do{
    j++;
    partsum += (double) sort[j];
  } while((partsum < rand) && (j < size - 1));
  return j;
}


/* fitness proportional selection of mates for GA */

int Roulette(IPTR pop, int popsize, Population *p)
{ 
  /* select a single individual by roulette wheel selection */
  double rand,partsum;
  int j;

  partsum = 0.0; j = 0;

  if(p->scaleFactor > 0.0) {
    rand = FRandom() * p->scaledSumFitness; 
    j = -1;
    do{
      j++;
      partsum += pop[j].scaledFitness;
    } while (partsum < rand && j < popsize - 1) ;
    return j;
  } else {
    rand = FRandom() * p->sumFitness; 
    j = -1;
    do{
      j++;
      partsum += pop[j].fitness;
    } while (partsum < rand && j < popsize - 1) ;
    return j;
  }
}

int RandomMate(IPTR pop, int popsize, Population *p)
{
  return Rnd(0, popsize - 1);
}

