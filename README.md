# genefer22
Generalized Fermat Prime search program

## About

**genefer22** is an [OpenMP®](https://www.openmp.org/) application on CPU and an [OpenCL™](https://www.khronos.org/opencl/) application on GPU.  
It performs a fast probable primality test for numbers of the form *b*<sup>2<sup>*n*</sup></sup> + 1 with [Fermat test](https://en.wikipedia.org/wiki/Fermat_primality_test).  
[genefer](https://doi.org/10.5334/jors.ca) was created by Yves Gallot in 2001. It has been extensively used by [PrimeGrid](https://www.primegrid.com/forum_forum.php?id=75) computing project.  

*genefer22* is a new application, created in 2022. It's going to replace *genefer*.  
The test is validated with [Gerbicz - Li](https://www.mersenneforum.org/showthread.php?t=22510) error checking and a proof is generated with ([Pietrzak - Li](https://eprint.iacr.org/2018/627.pdf)) algorithm. Thanks to the Verifiable Delay Function, distributed projects run at twice the speed of double-checked calculations.  

Any number of the form *b*<sup>2<sup>*n*</sup></sup> + 1 such that 2 &le; *b* < 2,000,000,000 and 12 &le; *n* &le; 22 can be tested on GPU.  
On CPU, the limit is fuzzy and is *b* ~ 500M (*n* = 12), 380M (*n* = 13), 290M (*n* = 14), 220M (*n* = 15), 160M (*n* = 16), 125M (*n* = 17), 94M (*n* = 18), 71M (*n* = 19), 54M (*n* = 20), 41M (*n* = 21), 31M (*n* = 22).  

## Build

The stage of development of *genefer22* is Release Candidate.  

This version is compiled with gcc 11 and was tested on Windows and Linux (Ubuntu).  

## TODO

 - store checkpoints into GPU memory for small GFNs
