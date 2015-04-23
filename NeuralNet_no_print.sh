#!/bin/sh
rm -rf logs/
echo "Deleted the old logs"
mkdir logs
echo "Creating new logs"
python NeuralNet_no_print.py -wlimit 0 -dn 10 -word 50 -window 5 -arguments -iter 5 -in input/in.txt
