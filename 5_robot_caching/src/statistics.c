#include <stdio.h>
#include "type.h"

#include <math.h>

void Scalepop(IPTR pop, Population *p);

void Statistics(IPTR pop, Population *p)
{ /* calculate population stats */
  int  j;
  register IPTR pj;
  double sum;
  
  sum = pop[0].fitness;
  p->max = sum;
  p->min = sum;
  p->maxi = p->mini = 0;
  for(j = 1; j < p->popsize;j++){
    pj = &(pop[j]);
    sum += pj->fitness; 
    if (p->max < pj->fitness) {
      p->max = pj->fitness;   p->maxi = j;
    }
    if (p->min > pj->fitness){
      p->min = pj->fitness;   p->mini = j;
    }
  }
  p->sumFitness = sum;
  p->avg = sum / (double) p->popsize;
  if(p->bigMax < p->max) {
    p->bigMax = p->max; 
    p->bigGen = p->generation; 
    p->bigInd = p->maxi;
  }
  Scalepop(pop, p);
  p->smax = p->scaleConstA * p->max + p->scaleConstB;
  p->smin = p->scaleConstA * p->min + p->scaleConstB;
}
