#include <stdio.h>
#include "type.h"
#include "select.h"

double EvalPopulation(IPTR pj, int start, int end, Population *p);
void   Halve(IPTR o, IPTR n, Population *p);
int Generation0(IPTR oldpop, IPTR newpop, int t, Population *p, Functions *f) 
{
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
  EvalPopulation(newpop, 0, p->popsize, p);
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

  EvalPopulation(oldpop, p->popsize, p->popsize * p->lambda, p);
  Halve(oldpop, newpop, p);

  return 0;

}

