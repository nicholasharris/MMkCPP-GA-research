CC = gcc

HDRS =  type.h defaults.h case.h random.h utils.h select.h finders.h \
	dist.h ox.h 

SRCS =  xover.c gen.c cigar.c report.c \
	statistics.c  random.c  select.c utils.c halve.c scale.c \
	finders.c hms.c case.c dist.c\
	pmx.c evalpop.c ox.c lcs.c

OBJS =  xover.o gen.o cigar.o report.o \
	statistics.o  random.o  select.o utils.o halve.o scale.o \
	finders.o hms.o case.o  dist.o\
	pmx.o evalpop.o ox.o lcs.o

APPHDRS = app.h init.h greedy.h

APPSRCS = app.c greedy.c init.c 
# greedy must be listed here since it uses app specific information
# greedy xover calculates distances between cities
APPOBJS = app.o greedy.o init.o

TESTHDRS = testapp.h testinit.h

TESTSRCS = testapp.c testinit.c

TESTOBJS = testapp.o testinit.o

TESTINFILE = testinfile

INFILE  = infile eil76.tsp berlin52.tsp eil51.tsp lin105.tsp lin318.tsp run.sh

BASENAME	= cigartsp-1.1.tgz

CFLAGS =  -O2 -Wall
CLIBS = -lm 

cigar	: $(OBJS) $(APPOBJS)
	$(CC) -o cigar $(OBJS) $(APPOBJS) $(CLIBS)

cigartest	: $(OBJS) $(TESTOBJS)
	$(CC) -o cigartest $(OBJS) $(TESTOBJS) $(CLIBS)

clean	: 
	rm $(OBJS) $(APPOBJS)
	touch $(HDRS) $(APPHDRS)

dist	:
	tar -cvzf $(BASENAME) $(HDRS) $(APPHDRS) $(SRCS) $(APPSRCS) $(INFILE) Makefile $(TESTHDRS) $(TESTSRCS) $(TESTINFILE)
	cp $(BASENAME) /staff/sushil/classes/gas/code/cigar/

$(OBJS)	: $(HDRS) $(APPHDRS)

$(APPOBJS)	:	$(APPHRDS)
