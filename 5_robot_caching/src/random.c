/* randgen.c => contains random number generator and related utilities
   including advance_random, warmup_random, random, randomize
*/

#include <stdio.h>

#include "type.h"


/* GLOBAL VARIABLES */

double oldrand[56];     /* array of 55 random numbers */
int    jrand;          /* current random */

/* FUNCTIONS */

void advance_random()   /* create next batch of 55 random numbers */
{
  int j1;
  double new_random;

/*  printf("Entering Advance Random\n");*/
  for (j1=1;j1<=24;j1++){
    new_random = oldrand[j1] - oldrand[j1+31];
    if (new_random < 0.0){
      new_random = new_random + 1.0;
    }
    oldrand[j1] = new_random;
  }
  for (j1=25;j1<=55;j1++){
    new_random = oldrand[j1] - oldrand[j1-24];
    if (new_random < 0.0){
      new_random = new_random + 1.0;
    }
    oldrand[j1] = new_random;
  }
  /*  printf("leaving Advance Random\n");*/
}

void warmup_random(double random_seed)
{
  int    j1, ii;
  double new_random, prev_random;
  
  oldrand[55] = random_seed;
  new_random  = 0.0000000009;
  prev_random = random_seed;
  for (j1=1;j1<=54;j1++){
    ii = (21*j1) % 55;
    oldrand[ii] = new_random;
    new_random  = prev_random - new_random;
    if (new_random < 0.0){
      new_random += 1.0;
    }
    prev_random = oldrand[ii];
  }
  advance_random();
  advance_random();
  advance_random();
  jrand = 0;
}

double FRandom()
{
  /* Fetch a single random number between 0.0 and 1.0 - Substractive Method 
     See Knuth, D. (1969), v.2 for details */
  
  jrand++;
  if (jrand > 55){
    jrand = 1;
    advance_random();
  }
  return (oldrand[jrand]);
}

void Randomize(double seed)  /* Get seed number for random and start it up */
{
  
  if(seed < 0.0 || seed > 1.0) {
    fprintf(stderr, "ERROR: seed random number must be betwwn 0 and 1");
    exit(1);
  }
  warmup_random(seed);
}


int Flip(double prob)
{/* Flip a biased coin...true if heads */
  
  if (prob == 1.0) {
    return 1;
  } else {
    if (FRandom() <= prob) {
      return 1;
    }else {
      return 0;
    }
  }
}

int Rnd(int low, int high)
{ /* pick a random int between low and high */
  double fr;
  int i;
  
  fr = FRandom();
  
  if (low >= high) {    
    i = low;
  } else {
    i = ((int)(fr * (double)(high - low + 1)) + low);
    if (i > high) i = high;
  }
  
  
  return i;
}


