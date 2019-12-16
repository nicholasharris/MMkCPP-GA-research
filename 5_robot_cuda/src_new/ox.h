
#ifndef __OX_H__
#define __OX_H__

int FindCity(ChromType city, ChromType *tour, int lchrom);
void SwapMutate(ChromType *ci1, ChromType * ci2, int lchrom, float pMut);

void PMX(IPTR p1, IPTR p2, IPTR c1, IPTR c2, Population *p);
void Copy(ChromType *child, ChromType *parent, int lchromx);
void SwapCity(ChromType *, ChromType *, int lchrom);

void InvertMutate(ChromType *ci, int lchrom, float pMut);
void Invert(ChromType *ci, int start, int end);

void SlideMutate(ChromType *c1, int lchrom, float pMut);
void Slide(ChromType *c1, int pick, int target, int lchrom);

void DoOx(IPTR p1, IPTR p2, IPTR c1, IPTR c2, Population *p);
int Member (int key, int *sortedarray, int size);
void TourPrint(FILE *fp, IPTR pj, char *name);

#endif

