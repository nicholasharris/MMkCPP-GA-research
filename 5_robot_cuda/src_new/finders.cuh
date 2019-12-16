#ifndef __FINDERS_H__
#define __FINDERS_H__

void FindNFurthest(double *sort, int *rank, int size, int howmany);

void FindNFurthProbable(double *sort, int *rank, int size, int howmany);

void FindNClosest(double *sort, int *rank, int size, int howmany);

void FindNCloseProbable(double *sort, int *rank, int size, int howmany);

void FindNPopRandom(double *sort, int *rank, int size, int howmany);

void FindNRanRandom(double *sort, int *rank, int size, int howmany);

void FindNWorst(INDIVIDUAL *p, int *rank, int size, int howmany);

void FindNBest(INDIVIDUAL *p, int *rank, int size, int howmany);

#endif
