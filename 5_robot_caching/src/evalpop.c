
#include "type.h"

double Eval(IPTR pj);

void EvalPopulation(IPTR pop, int start, int end, Population *p)
{
  int i;
  IPTR pj;

  if(p->maximize){
    for(i = start; i < end; i++){
      pj = &pop[i];
      pj->fitness = Eval(pj);
    } 
  }else {
    for(i = start; i < end; i++){
      pj = &pop[i];
      pj->objfunc = Eval(pj);
      pj->fitness = p->maxConst - pj->objfunc;
    }
  }
  return;

}

