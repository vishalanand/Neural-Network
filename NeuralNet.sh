#!/bin/sh
rm -rf logs/
echo "Deleted the old logs"
mkdir logs
echo "Creating new logs"
python NeuralNet.py -wlimit 0 -dn 4 -word 2 -window 3 -arguments -in input/in.txt
