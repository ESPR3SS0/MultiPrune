
# MutliPrune! 

This repo is an extension of: https://github.com/UCdasec/TinyRadio.git

Check out this repository for more information regarding the models and dataset.


### Obj 

This Repo implements (naively) two new pruning methods. Specifically, in 
addition to l2-norm pruning there is now:
1. Leverage score pruning 
2. Volume score (based) pruning 


This application serves as an example to illustrate the differences between 
the different metrics that are used to measure the importance of a node in 
a neural network. 


### Conclusion... 

Breifly 
1. Leverage sampling resulted in pruning very similar to that of l2-norm 
2. Volume sampling resulted in the most agressive pruning technique. 
