#include <math.h>
#include <stdlib.h>
#include "type.cuh"
#include "dist.cuh"
#include "utils.cuh"

void HelpPrint(FILE *fp, ChromType *ch, int len, char *name);
void SetInserted(ChromType *tmplate, ChromType *n, 
		 ChromType *p1, ChromType *p2,  
		 int tlen, int p1len, int p2len, int chromLen);
int Member (int key, int *array, int size);
double HamDist(ChromType *p1, ChromType *p2, int size1, int size2, ChromType **tmplate)
{
/*  find the hamming ditance between two binary strings */
  int dist,i;

  dist = 0;
  if(size1 > size2) SwapInt(&size1, &size2);
  for(i = 0; i < size1; i++) {
    dist += (p1[i] == p2[i] ? 0 : 1);
  }
  return (double) dist;
}

double Euclidean(ChromType *p1, ChromType *p2, int size1, int size2, ChromType **tmplate)
{
  double dist, tmp;
  int i;

  dist = 0.0;
  if(size1 > size2) SwapInt(&size1, &size2);
  for(i = 0; i < size1; i++){
    tmp =  (p1[i] - p2[i]);
    dist += tmp * tmp;
  }
  return sqrt((double) dist);
}

/* LCS is in lcs.c */
double LCS(ChromType *p1, ChromType *p2, int size1, int size2,
		 ChromType **tmplate)
{
  int length, dist;
  length = DoLcs(p1, p2, size1, size2, *tmplate);
  dist = abs(size1 - length);
  //  fprintf(stdout, "LCS: length: %i\n", length);
  //  HelpPrint(stdout, *tmplate, length, "LCStmp" );
  FixTemplate(tmplate, length, p2, size2, p1, size1);

  return ((double) dist);
}
  

/* this is making me think we should have  a separate cigar version for
   sequential representations ??   */ 

void FixTemplate(ChromType **tmplate, int tlen, ChromType *p2, int p2len, 
		 ChromType *p1, int chromLen)
     /* I want tmplate to be the right size for this problem (chromLen)
	Note length of template MUST be <= chromLen */
{
  int i;
  ChromType *n, *p1Copy, *p2Copy;
  ChromType *pt;

  pt = *tmplate;
  n = (ChromType *) calloc ((size_t) chromLen, sizeof(ChromType));
  p1Copy = (ChromType *) calloc ((size_t) chromLen, sizeof(ChromType));
  p2Copy = (ChromType *) calloc ((size_t) p2len, sizeof(ChromType));


  for(i = 0; i < tlen; i++){
    n[i]= pt[i];
  }
  for(i = 0; i < chromLen; i++){
    p1Copy[i] = p1[i];
  }
  for(i = 0; i < p2len; i++){
    p2Copy[i] = p2[i];
  }

  if(p2len > chromLen){
    SetInserted(pt, n, p2Copy, p1Copy, tlen, p2len, chromLen, chromLen);
  } else {
    SetInserted(pt, n, p1Copy, p2Copy, tlen, chromLen, p2len, chromLen);
  }
  free(*tmplate);
  *tmplate = n;    

  free(p1Copy);
  free(p2Copy);

}

void SetInserted(ChromType *tmplate, ChromType *n, 
		 ChromType *p1, ChromType *p2,  
		 int tlen, int p1len, int p2len, int chromLen)
{
  int i, j;
  for(i = 0; i < p2len; i++){
    n[i] = p2[i];
  }
  for(i = 0; i < tlen; i++){
    //    Remove(tmplate[i], p2, p2len);
    Remove(tmplate[i], p1, p1len);
  }

  for(i = 0; i < chromLen; i++){
    if(Member(n[i], tmplate, tlen) < 0){
      j = 0;
      while(((n[i] = GetRemoveUnused(p1, p1len)) >= chromLen) 
	    && (j++ < chromLen));
    }
  }

  //  HelpPrint(stdout, n, chromLen, "Fixn");
  //  HelpPrint(stdout, p1, chromLen, "Fixp 1");
  //  HelpPrint(stdout, *tmplate, tlen, "Fixtpl");

}

void HelpPrint(FILE *fp, ChromType *ch, int len, char *name)
{
  int i;
  fprintf(fp, "%s :", name);
  for(i = 0; i < len; i++){
    fprintf(fp, " %i", ch[i]);
  }
  fprintf(fp, "\n");
}



void FixCopiedIndividual(IPTR pj, Population *p)
{
  int i;
 
  pj->chromLen = p->chromLength;
  free(pj->chrom);
  pj->chrom = (ChromType *) calloc ((size_t) pj->chromLen, sizeof(ChromType));
  for(i = 0; i < pj->chromLen; i++){
    pj->chrom[i] = pj->backup[i];
  } /* copying fixed individual to chrom */
}

