
#include "type.cuh"

void FindCoeffs(IPTR pop, Population *p);

void Scalepop(IPTR pop, Population *p)
{ 
  /* linearly scale the population */
  IPTR pj;
  int i;
  
  FindCoeffs(pop, p);

  p->scaledSumFitness = 0.0;
  for(i = 0; i < p->popsize; i++){
    pj = &pop[i];
    pj->scaledFitness = p->scaleConstA * pj->fitness + p->scaleConstB;
    p->scaledSumFitness += pj->scaledFitness;
  }
}

void FindCoeffs(IPTR pop, Population *p)
{
  /* find coeffs scale_constA and scale_constB for linear scaling according to 
     f_scaled = scale_constA * f_raw + scale_constB */  

  double d;

  if(p->min > (p->scaleFactor * p->avg - p->max)/
     (p->scaleFactor - 1.0)) { /* if nonnegative smin */
    d = p->max - p->avg;
    p->scaleConstA = (p->scaleFactor - 1.0) * p->avg / d;
    p->scaleConstB = p->avg * (p->max - (p->scaleFactor * p->avg))/d;
  } else {  /* if smin becomes negative on scaling */
    d = p->avg - p->min;
    p->scaleConstA = p->avg/d;
    p->scaleConstB = -p->min * p->avg/d;
  }
  if(d < 0.00001 && d > -0.00001) { /* if converged */
    p->scaleConstA = 1.0;
    p->scaleConstB = 0.0;
  }
}


