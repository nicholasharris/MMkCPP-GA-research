#include <stdio.h>
#include <stdlib.h>

#include "type.h"
#include "utils.h"
#include "random.h"
#include "ox.h"

/* chromosome type must be int for sequencing problems */
void OX(IPTR p1, IPTR p2, IPTR c1, IPTR c2, Population *p)
{
  IndividualCopy(p1, c1);
  IndividualCopy(p2, c2);

  if (!Flip(p->pCross)) {
    return;
  }

  DoOx(p1, p2, c1, c2, p);
  DoOx(p2, p1, c2, c1, p);

  /* need to do mutation seperately here */
  /*  SwapMutate(c1->chrom, c2->chrom, p->chromLength, p->pMut);*/
  InvertMutate(c1->chrom, p->chromLength, p->pMut);
  InvertMutate(c2->chrom, p->chromLength, p->pMut);
  SlideMutate(c1->chrom, p->chromLength, p->pMut);
  SlideMutate(c2->chrom, p->chromLength, p->pMut);

}

void DoOx(IPTR p1, IPTR p2, IPTR c1, IPTR c2, Population *p)
{
  int  i, j, lchrom;
  int xp1, fcity;
  int *empty, *filled;
  int nempty, nfilled;

  lchrom = p->chromLength;

  xp1 = Rnd(0, lchrom - 1); 
  /* remember that rnd returns a number between */
  /* low and high inclusive */

  nempty = lchrom - xp1;
  nfilled = xp1;
  empty = (int *) calloc ((size_t) nempty, sizeof(int));
  filled = (int *) calloc ((size_t) nfilled, sizeof(int));
  //  printf("xp1: %d, nempty: %d, nfilled; %d\n", xp1, nempty, nfilled);
  j = 0;
  //  printf("Empty:");
  for(i = xp1; i < lchrom; i++) {
    fcity = FindCity(p1->chrom[i], p2->chrom, lchrom); 
    empty[j++] = fcity;
    //    printf(" %d", fcity);
  }
  //  printf("\n");

  /* find out filled empty */
  for(i = 0, j = 0; i < lchrom && j < nfilled ; i++){
    if(Member(i, empty, nempty) < 0){
      filled[j++] = i;
    }
  }
  //  printf("Filled:");
  for(i = 0; i < xp1; i++){ //xp1 == nfilled
    c2->chrom[i] = p2->chrom[filled[i]];
    //    printf(" %d", filled[i]);
  }
  //  printf("\n");
  for(i = xp1; i < lchrom; i++){
    c2->chrom[i] = p1->chrom[i];
  }
  //  TourPrint(stdout, p1, "p1");
  //  TourPrint(stdout, p2, "p2");
  //  TourPrint(stdout, c2, "c2");

  free(empty);
  free(filled);

}


int Member (int key, int *array, int size)
{
  int i;
  for(i = 0; i < size; i++){
    if(key == array[i]) {
      return i;
    } 
  }
  return -1;
}




