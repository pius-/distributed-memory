#!/bin/bash

# simply script to check the output from slurm
# border values should be 1 and the array should be symmetrical

directory=$1
rows=$2

for filename in $directory/*.txt; do

# array starts from row 5
startrow=5;
endrow=$(($startrow + $rows - 1))
midrow=$(($startrow + $rows / 2))

## Check all 1 in first row and first column
# symmetry check will ensure last row and last column is also 1
ok1=`awk -v start=$startrow 'NR == startrow { for (i=1; i<=NF; i++) if ($i != 1) print "NO"}' $filename`
ok2=`awk -v start=$startrow -v end=$endrow 'NR>=start && NR<=end && $1 != 1 {print "NO"}' $filename`


## Check symmetry
# splits the 1000 dimension array into two sections,
# one with first 500 lines using sed
# second with last 500 lines using sed
# second section rows are reversed using tac
# second sections columns are reversed using awk
# check symmetry using diff

ok3=`diff --ignore-space-change <(sed -n "$startrow,$(($midrow - 1)) p" $filename) \
<(sed -n "$midrow,$endrow p" $filename | tac \
| awk '{ for (i=NF; i>1; i--) printf("%s ", $i); print $1; }')`


if [[ $ok1 ]]; then
    echo $filename "FAILED: first row check"
fi

if [[ $ok2 ]]; then
    echo $filename "FAILED: first column check"
fi

if [[ $ok3 ]]; then
    echo $filename "FAILED: symmetry check"
fi

if [[ -z "${ok1}${ok2}${ok3}" ]]; then
    echo $filename "OK"
fi

done
