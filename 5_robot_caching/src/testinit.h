void Usage(void);

void GetInputParameters(Population *p, Functions *f);
void InitFiles(Population *p);
void InitGooguReport(IPTR op, Population *p);
void Statistics(IPTR pop, Population *p);
void Initreport(Population *p);
void NPointCrossover(IPTR p1, IPTR p2, IPTR c1, IPTR c2, Population *p );
void Block(IPTR p1, IPTR p2, IPTR c1, IPTR c2, Population *p );
void PMX(IPTR, IPTR, IPTR, IPTR, Population *p);
void M2S(IPTR, IPTR, IPTR, IPTR, Population *p);
void OX(IPTR p1, IPTR p2, IPTR c1, IPTR c2, Population *p);

void InitGA(Population *p, Functions *f);

void InitPhenoPrint(IPTR op, char *phenotypeFile, Population *p);
void PhenoPrint(IPTR pj, FILE *fp);

void AppInitChrom(IPTR pj);
void AppInit(char *appInfile, Population *p);

void RawStat(FILE *fp, IPTR pop);

int Generation0(IPTR oldpop, IPTR newpop, int t, Population *p, Functions *f) ;
int CHC(IPTR oldpop, IPTR newpop, int t, Population *p, Functions *f);
int Roulette(IPTR pop, int popsize, Population *p);
int RandomMate(IPTR pop, int popsize, Population *p);

void Report(int gen, IPTR pop, Population *p);

