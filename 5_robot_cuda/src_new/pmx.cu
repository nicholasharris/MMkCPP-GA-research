#include <stdio.h>
#include <cstdlib>
#include "type.cuh"
#include "utils.cuh"
#include "random.cuh"

int FindCity(ChromType city, ChromType *tour, int lchrom);
void SwapMutate(ChromType *ci1, ChromType * ci2, int lchrom, float pMut);

void PMX(IPTR p1, IPTR p2, IPTR c1, IPTR c2, Population *p);
void Copy(ChromType *child, ChromType *parent, int lchromx);
void SwapCity(ChromType *, ChromType *, int lchrom);

void InvertMutate(ChromType *ci, int lchrom, float pMut);
void Invert(ChromType *ci, int start, int end);

void SlideMutate(ChromType *c1, int lchrom, float pMut);
void Slide(ChromType *c1, int pick, int target, int lchrom);

/* chromosome type must be int for sequencing problems */
void Copy(ChromType *child, ChromType *parent, int lchrom)
{
  int i;
  for(i = 0; i < lchrom; i++){
    child[i] = parent[i];
  }
}

void PMX(IPTR p1, IPTR p2, IPTR c1, IPTR c2, Population *p)
{
  int  i;
  int xp1, xp2, fcity;

  IndividualCopy(p1, c1);
  IndividualCopy(p2, c2);

  if (!Flip(p->pCross)) {
    return;
  }

  /*   ncross++; */
  xp1 = Rnd(0, p->chromLength - 1); /* remember that rnd returns a number between */
  xp2 = Rnd(0, p->chromLength - 1); /* low and high inclusive */

  if(xp2 < xp1) SwapInt(&xp1, &xp2);

  for(i = xp1; i <= xp2; i++) {
    /*<= since both are guaranteed to be in range */

    fcity = FindCity(p1->chrom[i], p2->chrom, p->chromLength); 
    /* making c2 */
    SwapChromType(& c2->chrom[fcity], &c2->chrom[i]); 
    /* SwapInt aka SwapCity */

    fcity = FindCity(p2->chrom[i], p1->chrom, p->chromLength); 
    /* making c1 */
    SwapChromType(&c1->chrom[fcity], & c1->chrom[i]);
  }
  
  /* need to do mutation seperately here */
  //SwapMutate(c1->chrom, c2->chrom, p->chromLength, p->pMut);
  InvertMutate(c1->chrom, p->chromLength, p->pMut);
  InvertMutate(c2->chrom, p->chromLength, p->pMut);
  SlideMutate(c1->chrom, p->chromLength, p->pMut);
  SlideMutate(c2->chrom, p->chromLength, p->pMut);

}

void SwapMutate(ChromType *ci1, ChromType * ci2, int lchrom, float pMut)
{
  int i;
  for(i = 0; i < lchrom; i++){
    if(Flip(pMut)){
      SwapChromType(&ci1[Rnd(0, lchrom-1)], &ci1[Rnd(0, lchrom-1)]);
    }
    if(Flip(pMut)){
      SwapChromType(&ci2[Rnd(0, lchrom-1)], & ci2[Rnd(0, lchrom-1)]);
    }
  }
}

void InvertMutate(ChromType *ci, int lchrom, float pMut)
{
  int start, end;
  if(Flip(pMut)){
    start = Rnd(0, lchrom - 1);
    end = Rnd(0, lchrom - 1);
    if(start > end) SwapInt(&start, &end);
    Invert(ci, start, end);
  }
}

void SlideMutate(ChromType *c1, int lchrom, float pMut)
{
  int start, end;
  if(Flip(pMut)){
    start = Rnd(0, lchrom - 1);
    end = Rnd(0, lchrom - 1);

    Slide(c1, start, end, lchrom);
  }
}

void Slide(ChromType *c1, int pick, int target, int lchrom)
{
  int i;
  ChromType tmp;
  tmp = c1[pick];

  if(pick < target) {
    for(i = pick; i < target; i++){
      c1[i] = c1[i+1];
    }
    c1[target] = tmp;
  } else if (target < pick) {
    for(i = pick; i > target; i--){
      c1[i] = c1[i-1];
    }
    c1[target] = tmp;
  }
}

void Invert(ChromType *ci, int start, int end)
{
  int i;
  int mid = (start + end)/2;
  for(i = start; i < mid; i++){
    SwapChromType(&ci[i], &ci[end - (i - start)] );
  }
}

int FindCity(ChromType city, ChromType *tour, int lchrom)
{
  int i;
  for(i = 0; i < lchrom; i++) {
    if(city == tour[i]) return i;
  }
  fprintf(stderr, "FindCity: Couldn't find city %d\n", city);
  /*  PrintTour(tour, stderr); */
  exit(1);
}




