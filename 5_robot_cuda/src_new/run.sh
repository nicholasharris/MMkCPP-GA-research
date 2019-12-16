#! /bin/sh

# -i input file name
# -a application data file name
# -M minimization problem
# -s random seed between 0 and 1, needed to seed the random number generator
#
./cigar -i infile -a eil76.tsp -M -s 0.47657
#
# ./cigar -i infile -a eil76.tsp -M -s 0.47657 > /dev/null 
# will redirect most output to /dev/null

