#include <stdio.h>
#include <stdlib.h>

#include "type.h"
#include "utils.h"
#include "random.h"

#include "greedy.h"

/* chromosome type must be int for sequencing problems */
void Greedy(IPTR p1, IPTR p2, IPTR c1, IPTR c2, Population *p)
{
  IndividualCopy(p1, c1);
  IndividualCopy(p2, c2);

  if (!Flip(p->pCross)) {
    return;
  }

  DoGreedy(p1, p2, c1, c2, p);
  DoGreedy(p2, p1, c2, c1, p);
  /*
  TourPrint(stdout, p1, "p1");
  TourPrint(stdout, p2, "p2");
  TourPrint(stdout, c1, "c1");
  TourPrint(stdout, c2, "c2");
  */
  /* need to do mutation seperately here */
  SwapMutate(c1->chrom, c2->chrom, p->chromLength, p->pMut);
  //  InvertMutate(c1->chrom, p->chromLength, p->pMut);
  //  InvertMutate(c2->chrom, p->chromLength, p->pMut);
  //  SlideMutate(c1->chrom, p->chromLength, p->pMut);
  //  SlideMutate(c2->chrom, p->chromLength, p->pMut);

}

void DoGreedy(IPTR p1, IPTR p2, IPTR c1, IPTR c2, Population *p)
{
  int  i, lchrom;
  ChromType *unused;
  int p1C1Loc, p1C2Loc, p2C1Loc, p2C2Loc;

  lchrom = p->chromLength;
  unused = (ChromType *) calloc ((size_t) lchrom, sizeof(ChromType));
  Shuffle((int *) unused, lchrom);

  c1->chrom[0] = p1->chrom[0];
  Remove(c1->chrom[0], unused, lchrom);
  for(i = 1; i < lchrom; i++) {
    p1C1Loc = FindCity(c1->chrom[i-1], p1->chrom, lchrom);
    p1C2Loc = (p1C1Loc >= lchrom - 1 ? 0 : p1C1Loc + 1);
    p2C1Loc = FindCity(c1->chrom[i-1], p2->chrom, lchrom);
    p2C2Loc = (p2C1Loc >= lchrom - 1 ? 0 : p2C1Loc + 1);
    if(TSPDist(c1->chrom[i-1], p1->chrom[p1C2Loc])  >= 
       TSPDist(c1->chrom[i-1], p2->chrom[p2C2Loc])) {
      SetC(i, c1, p1, p2, p1C2Loc, p2C2Loc, lchrom, unused);
    } else {
      SetC(i, c1, p2, p1, p2C2Loc, p1C1Loc, lchrom, unused);
    }
  }
      
  free(unused);
}

void SetC(int i, IPTR c1, IPTR p1, IPTR p2, 
	  int p1C2Loc, int p2C2Loc, int lchrom, ChromType *unused)
{
  if (Member(p1->chrom[p1C2Loc], c1->chrom, i) < 0) {
    c1->chrom[i] = p1->chrom[p1C2Loc];
    Remove(p1->chrom[p1C2Loc], unused, lchrom);
  } else if (Member(p2->chrom[p2C2Loc], c1->chrom, i) < 0) {
    c1->chrom[i] = p2->chrom[p2C2Loc];
    Remove(p2->chrom[p2C2Loc], unused, lchrom);
  } else {
    c1->chrom[i] = GetRemoveUnused(unused, lchrom);
  }
}
      


