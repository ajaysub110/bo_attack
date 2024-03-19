#!/bin/bash

for i in {16..49}
do
   python run_bayesopt_attack.py -f='mnist' -m='GP' -nitr=200 -rd='BILI' -ld=196 -i=$i -ntg=9 &
done

wait
