#include <stdio.h>
#include <stdlib.h>

#include "type.h" 
#include "utils.h"
#include "random.h"

void Mutate(IPTR pj, float pMut);
int MuteX(int pa, int pb, int flag);
float pMut;
float pCross;

void NPointCrossover(IPTR p1, IPTR p2, IPTR c1, IPTR c2, Population *p )
{
/* p1,p2,c1,c2,m1,m2,mc1,mc2 */
  ChromType *pi1,*pi2,*ci1,*ci2;
  int i, k;
  int *xp;
  int lchrom;

  lchrom = p->chromLength;
  pMut = p->pMut;
  pCross = p->pCross;

  pi1 = p1->chrom;
  pi2 = p2->chrom;
  ci1 = c1->chrom;
  ci2 = c2->chrom;

  
  IndividualCopy(p1, c1);
  IndividualCopy(p2, c2);

  /* only Mutation */
  if(!Flip(pCross)) {
    Mutate(c1, pMut);
    Mutate(c2, pMut);
    return;
  }
  /* Uniform Crossover */
  if(p->nXPoints == lchrom){
    for(i = 0; i < lchrom; i++){
      ci1[i] = (Flip(0.5) ? pi1[i] : pi2[i]);
      ci2[i] = (Flip(0.5) ? pi1[i] : pi2[i]);
    }
    Mutate(c1, pMut);
    Mutate(c2, pMut);
    return;
  }

  /* N point crossover */
  xp = (int *) calloc (lchrom, sizeof(int));
  for(i = 1; i < p->nXPoints; i++){
    xp[Rnd(0, lchrom - 1)] = 1;
  }
  k = 0;
  for(i = 0; i < lchrom; i++){
    if(xp[i] == 1) {
      k++; 
    }
    ci1[i] = MuteX(pi1[i], pi2[i], k%2);
    ci2[i] = MuteX(pi2[i], pi1[i], k%2);
  }
  /* end crossover */
  Mutate(c1, pMut);
  Mutate(c2, pMut);
  free(xp);
}

int MuteX(int pa, int pb, int flag)
{
  return (flag ? pa : pb );

}

void Mutate(IPTR pj, float pMut)
{
  int i;
  for(i = 0; i < pj->chromLen; i++){
    pj->chrom[i] = (Flip(pMut) ? 1 - pj->chrom[i] : pj->chrom[i]);
  }
}


  
