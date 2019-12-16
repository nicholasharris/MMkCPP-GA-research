#ifndef  __DIST_H__
#define  __DIST_H__

#define HAMD 0
#define EUCD 1
#define LCSD 2

double Euclidean(ChromType *p1, ChromType *p2, 
		 int size1, int size2, ChromType **tmplate);
double HamDist(ChromType *p1, ChromType *p2, 
	       int size1, int size2, ChromType **tmplate);

double LCS(ChromType *p1, ChromType *p2, 
	   int size1, int size2,
	   ChromType **tmplate);


double DoLcs(ChromType *x, ChromType *y, int m, int n, 
	     ChromType *tmplate);

void FixTemplate(ChromType **tmplate, int tlen, ChromType *p2, int p2len, 
		 ChromType *p1, int chromLen);
void FixCopiedIndividual(IPTR pj, Population *p);

#endif
