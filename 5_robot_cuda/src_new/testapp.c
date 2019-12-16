#include <stdio.h>
#include "type.h"

double Eval(IPTR pj)
{
  int i;
  double sum;

  sum = 0.0;
  for(i = 0; i < pj->chromLen; i++){
    sum += pj->chrom[i];
  }
  return sum;
}

void AppInitChrom(IPTR pj)
{
  return;
}

void AppInit(char *appInfile)
{


}

void PhenoPrint(FILE *fp, IPTR pj, Population *p)
{

  /************
  for(i = 0; i < pj->chromLen; i++){
    fprintf(stdout, "%1i", pj[p->maxi].chrom[i]);
  }
  fprintf(stdout, "\n");
  **************/
}

void InitPhenoPrint(IPTR pj, char *fname, Population *p)
{

}
