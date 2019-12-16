
#ifndef __GREEDY_H__
#define __GREEDY_H__

int FindCity(ChromType city, ChromType *tour, int lchrom);
void SwapMutate(ChromType *ci1, ChromType * ci2, int lchrom, float pMut);

void SwapCity(ChromType *, ChromType *, int lchrom);

void InvertMutate(ChromType *ci, int lchrom, float pMut);
void Invert(ChromType *ci, int start, int end);

void SlideMutate(ChromType *c1, int lchrom, float pMut);
void Slide(ChromType *c1, int pick, int target, int lchrom);

void DoGreedy(IPTR p1, IPTR p2, IPTR c1, IPTR c2, Population *p);
int Member (int key, int *sortedarray, int size);
void TourPrint(FILE *fp, IPTR pj, char *name);

void SetC(int i, IPTR c1, IPTR p1, IPTR p2, 
	  int p1C2Loc, int p2C2Loc, int lchrom, ChromType *unused);

double TSPDist(ChromType c1, ChromType c2);
#endif
