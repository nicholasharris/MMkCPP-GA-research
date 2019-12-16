#ifndef __DEFAULTS_H__
#define __DEFAULTS_H__

#define SEED                  0.00201297

#define LAMBDA                1
#define POPSIZE               50
#define MAXGEN                100
#define CHROMLENGTH           50		
#define PCROSS                0.66
#define PMUT                  0.001
#define NXPOINTS              2
#define SCALEFACTOR           1.2

#define CROSSOVER             NPointCrossover
#define CURRENTGA             Generation0
#define FINDMATE              Roulette

#define OUTPUTFILE            "ofile"
#define CASEFILE              "cb"

#endif
