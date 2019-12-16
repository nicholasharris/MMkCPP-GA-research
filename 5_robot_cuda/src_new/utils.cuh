#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef _UTILS_H
#define _UTILS_H

#define TRUE                  (1 == 1)
#define FALSE                 (0 == 1)

void   strcreate(char **dest, char *src);
void   syserror(char *loc);
void   error(char *loc, char *message);
int    Flip(double prob);
void   SwapInt(int *, int *);
void   SwapChromType(ChromType *, ChromType *);
int    AddModulo(int lim, int x, int y);
void   skipline(FILE *fp);
void   Shuffle(int *deck, int size);
void   IndividualCopy(individual *from, individual *to);

__device__ double Eval(IPTR pj);

IPTR AllocateIndividuals(int howmany, int lchrom);

void DecToBin(int val, int size, ChromType *bits);
void CopyBits(ChromType *bits, ChromType *gene, int size);

void IntSort(int *array, int howmany);

void Remove(ChromType city, ChromType *unused, int lchrom);
ChromType GetRemoveUnused(ChromType *unused, int lchrom);

#endif /* _UTILS_H */
