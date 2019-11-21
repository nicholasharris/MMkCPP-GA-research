#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "type.h"
#include "utils.h"
#include "random.h"

IPTR AllocateIndividuals(int howmany, int lchrom)
{
  int i;
  IPTR tmp, pj ;
  
  tmp = (IPTR) malloc(sizeof(INDIVIDUAL) * howmany);
  if(tmp == NULL) 
    return NULL;
  for(i = 0; i < howmany; i++){
    pj = &(tmp[i]);
    pj->chrom = (ChromType *) malloc (sizeof(ChromType) * lchrom);
    if(pj->chrom == NULL) 
      return NULL;
    
    pj->backup = (ChromTypeB *) malloc (sizeof(ChromTypeB) * lchrom);
    if(pj->backup == NULL) 
      return NULL;
    pj->chromLen = lchrom;
  }
  return tmp;
}

void DeAllocateIndividuals(IPTR pj, int howmany)
{
  int i;
  if(pj == NULL) 
    return;
  for(i = 0; i < howmany; i++){
    free((pj + i)->chrom);
    free((pj + i)->backup);
    free((pj + i));
  }
  return;
}

void strcreate(char **dest, char *src) {
  if(*dest != NULL) {
    fprintf(stderr, "Strcreat: Dest should be set to NULL\n");
    exit(1);
  }

  *dest = (char *)malloc(sizeof(char) * strlen(src) + 1);
  if (!*dest) {
    perror("strcreate");
  }
  strcpy(*dest, src);
}


void Shuffle(int *deck, int size)
{
  int i;
  for(i = 0; i < size; i++){
    deck[i] = i;          /* Deal a deck ! */
  }
  for(i = 0; i < size/2; i++){ /* shuffle the deck */
    SwapInt(& deck[Rnd(0, size - 1)], & deck[Rnd(0, size - 1)]);
  }
}

double IntPow(int pow, int to)
{
  double prod;
  int i;

  prod = 1.0;
  for(i = 0; i < to; i++){
    prod *= (double) pow;
  }
  /*   printf("pow %d, to %d, prod %f\n", pow, to, prod);*/
  return prod;

}

void SwapInt(int *x, int *y)
{
  int tmp = *x;
  *x = *y;
  *y = tmp;
}

void SwapChromType(ChromType *x, ChromType *y)
{
  ChromType tmp = *x;
  *x = *y;
  *y = tmp;
}


void IndividualCopy(IPTR from, IPTR to)
{
  int i;

  to->chromLen = from->chromLen;
  /*  fprintf(stdout, "len %d\n", to->chromLen);*/
  for(i = 0; i < to->chromLen; i++){
    to->chrom[i] = from->chrom[i];
    to->backup[i] = from->backup[i];
    /*    fprintf(stdout, "i: %d; len %d\n", i, to->chromLen); */

  }
  to->fitness = from->fitness;
  to->scaledFitness = from->scaledFitness;
  to->objfunc = from->objfunc;
  to->dx = from->dx;
  to->dy = from->dy;
  to->cx = from->cx;

}
     
void skipline(FILE *fp)
{
  int ch; 
  while( (ch = fgetc(fp)) != '\n')
    if(ch == EOF)
      return;
  ch = fgetc(fp);
  if(ch == '#') {
    skipline (fp);
  } else {
    ungetc(ch, fp);
  }

}


int AddModulo(int lim, int x, int y)
{
  int tmp = x + y;
  if (tmp > lim) {
    return lim - y;
  } else {
    return tmp;
  }
}

void DecToBin(int val, int size, ChromType *bits)
{
  int i;
  for(i = 0; i < size; i++){
    bits[i] = val%2;
    val = val/2;
  }
}

void CopyBits(ChromType *bits, ChromType *gene, int size)
     /* copy from bits to gene */
{
  int i;
  for(i = 0; i < size; i++){
    gene[i] = bits[i];
  }
}

void IntSort(int *inarray, int howmany)
{
  int i, max, current;
  
  max = 0;
  for(current = 0; current < howmany - 1  ; current++) {
    max = current;
    /* ascending order */
    for(i = current + 1; i < howmany; i++) {
      if(inarray[i] < inarray[max]) {
	max = i;
      }
    }
    SwapInt(&inarray[current], &inarray[max]);
  }
}

void Remove(ChromType city, ChromType *unused, int lchrom)
{
  int i;
  for(i = 0; i < lchrom; i++){
    if(city == unused[i]) {
      unused[i] = -1;
      return;
    }
  }
}

ChromType GetRemoveUnused(ChromType *unused, int lchrom)
{
  int i;
  ChromType city;
  for(i = 0; i < lchrom; i++){
    if(unused[i] != -1){
      city = unused[i];
      unused[i] = -1;
      return city;
    }
  }
  fprintf(stderr, "GetRemovUnused: Could not find an Unused city\n");
  exit(2);
  return -1;
}

