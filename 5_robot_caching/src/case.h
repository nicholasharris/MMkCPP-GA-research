
#ifndef __CASE_H__
#define __CASE_H__

void LoadCases(IPTR pop, int gen, float frac, Population *p, Functions *h);
int GetBest(IPTR pop, int size);
int GetWorst(IPTR pop, int size);

void SaveCase(FILE *fp, IPTR pj, int gen, Population *p);
void ReadCase(FILE *fp, IPTR pj, Population *p);
void StoreNcases(char *ncfile, int ncases, int nCurrentCases);

int GetCases(IPTR pj, IPTR iCases, int howmany, Population *p, Functions *f);

void InitLoadCases(char *caseFile, IPTR pop, int gen, int perc, Population *p);

int FindNCases(char *ncfile);

void GetSetSeqChrom(FILE *fp, IPTR pj, Population *p, int len);

#endif

