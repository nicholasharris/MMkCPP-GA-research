#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "type.h"

#define UP     0
#define LEFT   1
#define UL     2
#define MAX(I, J)  ((I) > (J) ? (I) : (J))

typedef ChromType Pattern;


void LcsString(int **b, ChromType *x, int i, int j, ChromType *l, int li);
void MatPrint(int **cmat, int m, int n);

int EqualPat(ChromType *x, ChromType *y);

/* longest common subsequence implementation from CLR page 317 */
/* Sushil J. Louis */
/* x is newString and m is its length
   y is templateString and n is its length
*/

double DoLcs(ChromType *x, ChromType *y, int m, int n, 
	   ChromType *tmplate)
{
  int i, j;
  ChromType *tmp; /*  = "hello"; */
  int **cmat; /* DP matrix to track longest common subsequence */
  int **bmat; /* DP matrix to reconstruct string */
  int len;


  cmat = (int * *) malloc(sizeof(int *) * (m+1));
  for(i = 0; i < m + 1; i++){
    cmat[i] = (int *) malloc(sizeof(int) * (n+1));
  }

  bmat = (int * *) malloc(sizeof(int *) * (m+1));
  for(i = 0; i < m + 1; i++){
    bmat[i] = (int *) malloc(sizeof(int) * (n+1));
  }

  /* initialize cmat for main loop computation */
  for(i = 0; i < m + 1 ; i++){
    cmat[i][0] = 0;
  }
  for(j = 0; j < n + 1 ; j++){
    cmat[0][j] = 0;
  }

  /* main loop to calculate lcs */
  for(i = 1; i < m + 1; i++) {
    for(j = 1; j < n + 1; j++) {
      if(EqualPat(&x[i-1], &y[j-1])) {
	cmat[i][j] = cmat[i - 1][j - 1] + 1;
	bmat[i][j] = UL;
      } else if (cmat[i - 1][j] >= cmat[i][j - 1]) {
	cmat[i][j] = cmat[i - 1][j];
	bmat[i][j] = UP;
      } else {
	cmat[i][j] = cmat[i][j - 1];
	bmat[i][j] = LEFT;
      }
    }
  }

  /* tmp is temporary storage since we get the string in reverse */
  tmp = (ChromType *) malloc (sizeof(ChromType) * (m + 1));
  LcsString(bmat, x, m, n, tmp, 0);

  len = cmat[m][n]; /* length is the last entry in the matrix */

  /* reverse tmp to make the actual substring */

  for(i = 0; i < len ; i++){
    tmplate[i] = tmp[len - 1 - i];
  }


  for(i = 0; i < m + 1; i++){
    free(cmat[i]);
  }
  free(cmat);

  for(i = 0; i < m + 1; i++){
    free(bmat[i] );
  }
  free (bmat);
  free(tmp);

  return len;
}

void LcsString(int **b, ChromType *x, int i, int j, ChromType *l, int li)
{
  if(i == 0 || j == 0) {
    return;
  }
  if(b[i][j] == UL) {
      LcsString(b, x, i - 1, j - 1, l, li + 1);
      l[li] = x[i-1];
  } else if (b[i][j] == UP) {
      LcsString(b, x, i - 1, j, l, li);
  } else {
      LcsString(b, x, i, j - 1, l , li);
  }
}

void MatPrint(int **cmat, int m, int n)
{
  int i, j;

  for(i = 0; i < m ; i++){
    for(j = 0; j < n; j++){
      printf("%d ", cmat[i][j]);
    }
    printf("\n");
  }
}

int EqualPat(ChromType *x, ChromType *y)
{
  return (*x == *y);

}

