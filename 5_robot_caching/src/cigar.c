#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
//#include <unistd.h>
#include <errno.h>
//#include <ctime>
#include <time.h>

#include "type.h"
#include "case.h"

extern int errno;

void Statistics(IPTR, Population *p);
void Report(int gen, IPTR pop, Population *p);
void Initialize(int argc, char *argv[], Population *p, Functions *f);

void WritePid(char * pidFile);
void RmPidFile(char *pidFile);

//struct timespec start, finish;
//double elapsed;


int main(int argc, char *argv[])
{
	IPTR tmp;
	int foo; /* just a placeholder for a value that is not used */
	Population pop, *p;
	Functions funcs, *f;

	p = &pop;
	f = &funcs;

	p->generation = 0;

	Initialize(argc, argv, p, f);

	//WritePid(p->pidFile);

	while (p->generation < p->maxgen) {

		//start time recording
		//time_t start = time(clock());
	    //time(start);

		p->generation++;

		foo = f->CurrentGA(p->oldpop, p->newpop, p->generation, p, f);

		if (p->injectFraction > 0.0) {
			if ((p->generation%p->injectPeriod == 0)
				&& (p->generation <= p->injectStop)) {
				LoadCases(p->newpop, p->generation, p->injectFraction, p, f);
				/* printf("Loaded cases %d\n", (int) (loadPerc/100.0 * popsize));*/
			}
		}
		Statistics(p->newpop, p);
		Report(p->generation, p->newpop, p);

		//Record data (best individual at each gen) 
		FILE * dataFile;

		dataFile = fopen("myData.txt", "a");

		fprintf(dataFile, "%f\n", p->newpop[p->maxi].objfunc);

		fclose(dataFile);

		//Record best route
		FILE * routeFile;
		routeFile = fopen("myRoutes.txt", "a");
		PhenoPrint(routeFile, p->newpop, p);
		printf(routeFile, "\n");
		for(int i = 0; i < p->newpop->chromLen; i++)
			fprintf(routeFile, "%d, ", p->newpop->chrom[i]);
		fprintf(routeFile, "\n");
		fclose(routeFile);

		tmp = p->oldpop;
		p->oldpop = p->newpop;
		p->newpop = tmp;

		

		//record time
		//time_t end = time(clock());
		//double elapsed = difftime(end, start);
		
		//printf("\nTime Elapsed: %f\n", elapsed);
		//FILE * tFile;
		//tFile = fopen("myGenTimes.txt", "a");
		//fprintf(tFile, "%f\n", elapsed);
		//fclose(tFile);
	}
	if (p->nCurrentCases > 0) {
		p->nCases = FindNCases(p->nCFile);
		StoreNcases(p->nCFile, p->nCases, p->nCurrentCases);
	}
	//RmPidFile(p->pidFile);

	return 0;
}

void WritePid(char *fname)
{
	struct stat buf;
	int er;
	FILE *fp;

	er = stat(fname, &buf);
	if (!(er == -1 || errno == ENOENT)) {
		fprintf(stderr, "Lock file (%s) exists, Process running\n", fname);
		fprintf(stderr, "This process is exiting....\n");
		exit(1);
	}

	if ((fp = fopen(fname, "w")) == NULL) {
		fprintf(stderr, "Error in opening file %s for writing\n", fname);
		exit(2);
	}

	fprintf(fp, "%lu", getpid());
}

void RmPidFile(char *fname)
{
	unlink(fname);
}

