#include <stdio.h>
#include <stdlib.h>
#include "type.cuh"
#include "utils.cuh"

void FindNBest(INDIVIDUAL *p, int *rank, int size, int howmany);
void struct_cp(char *to, char *from, int size);
void InitAndShuffle(int *deck, int size);
void CopyIndividual(IPTR pj, IPTR qj);


void Halve(IPTR opop, IPTR npop, Population *p)
{

  int i;
  int *deck, *rank;
  
  deck = (int *) malloc (p->lambda * p->popsize * sizeof(int));
  rank = (int *) malloc (p->lambda * p->popsize * sizeof(int));

  for(i = 0;  i < p->popsize * p->lambda ; i++) {
    rank[i] = i;
  }
  FindNBest(opop, rank, p->lambda * p->popsize, p->popsize);

  Shuffle(deck, p->popsize);

  for(i = 0; i < p->popsize;i++){
    IndividualCopy(&opop[rank[i]], &npop[deck[i]]); 
    /* in utils.c */
    /*  struct_cp((char *) &pop[deck[i]], (char *) &opop[rank[i]], size);*/
  }

  free(rank);
  free(deck);

}
