
#include "type.cuh"
#include "utils.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "app.cu"

#include<cuda.h>

__device__ double Eval(IPTR pj, int* chroms, int index);


//parallelizes evaluation across many gpu threads
__global__
//void EvalPopulation(IPTR pop, int start, int end, int fitnesses[2])     //, Population *p)
void EvalPopulation(IPTR pop, int start, int end, int* chroms)
{
  int i;
  IPTR pj;

  //printf("start: %d\n", start);



  //select index based on thread/block id
  int index = start + blockIdx.x * blockDim.x + threadIdx.x;

  //printf("index: %d\n", index);
  //fitnesses[0] = 2;



  //calculate stride based on thread/block id
  int stride = blockDim.x * gridDim.x;
  //printf("Got here 1\n");
  //printf("Got here 1.5\n");
  //if(p->maximize){
	//  printf("Got here 2\n");
   // for(i = index; i < end; i += stride){
   if (index < end)
   {
      pj = &pop[index];

	  //int* chrom = (int*)malloc(pj->chromLen);

	  //printf("Got here 2.0, index: %d\n", index);

	  //for (int i = 0; i < pj->chromLen; i++)
	 // {
		  
	 // printf("chrom: %d\n", chroms[index*(pj->chromLen)]);
	 // }
   //
	  
	 // printf("got here1\n");
	  //printf("got here3\n");
	  //printf("chrom2: %d\n", pj->chrom[0]);
	  pj->objfunc = Eval(pj, chroms, index);
	  
	 // printf("Got here 2.1\n");
	 // printf("Got here 2.1, index: %d\n", index);
	  //pj->objfunc = Eval(pj);
	  //pj->fitness = Eval(pj);
	  //fitnesses[index] = 789;//Eval(pj);
	   //printf("Got here 2.2\n");
	   //pj->fitness = 80; // p->maxConst - pj->objfunc;
	   //printf("\nreturned fitness of: %d\n", pj->fitness);
    } 
 /* }else {
	  printf("Got here 3\n");
    for (i = index; i < end; i += stride) {
      pj = &pop[i];
      pj->objfunc = Eval(pj);
      pj->fitness = p->maxConst - pj->objfunc;
	  printf("\nreturned fitness of: %d", pj->fitness);
    }
  }*/
  //printf("Got here 4\n");
  return;

}

