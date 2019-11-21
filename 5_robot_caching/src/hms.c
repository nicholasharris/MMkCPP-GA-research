
#include "type.h"
#include <math.h>
#include <stdio.h>

double HamDist(ChromType *p1, ChromType *p2, int size);

void FindHammingStats(IPTR pop, Population *p)
{

  int i, n, j;
  int hDist;
  double sum, hamAvg, sdSum,hamVar;
  int seeling;
  double hamSD;

  seeling = p->popsize;

  hDist = 0;
  n = 0;
  j = 0;
  sum = 0.0;
  sdSum = 0.0;
  for(j = 0; j < seeling; j++){
    for(i = j+1; i < seeling; i++){
      hDist = HamDist((pop[j].chrom), (pop[i].chrom), p->chromLength);
      sum += hDist;
      sdSum += hDist * hDist;
      n++;
    }
  }

  hamAvg = sum/(double)n;
  hamVar = ((n * sdSum) - (sum * sum))/ (n * (n - 1));
  hamSD = sqrt(hamVar);

}


