# genefer22
Generalized Fermat Prime search program

## About

**genefer22** is an [OpenMP®](https://www.openmp.org/) application on CPU and an [OpenCL™](https://www.khronos.org/opencl/) application on GPU.  
It performs a fast probable primality test for numbers of the form *b*<sup>2<sup>*n*</sup></sup> + 1 with [Fermat test](https://en.wikipedia.org/wiki/Fermat_primality_test).  
[genefer](https://doi.org/10.5334/jors.ca) was created by Yves Gallot in 2001. It has been extensively used by [PrimeGrid](https://www.primegrid.com/forum_forum.php?id=75) computing project.  
genefer22 is a new application, created in 2022. It's going to replace *genefer*.  
The test is validated with [Gerbicz - Li](https://www.mersenneforum.org/showthread.php?t=22510) error checking and a proof is generated with ([Pietrzak - Li](https://eprint.iacr.org/2018/627.pdf)) algorithm. Thanks to the Verifiable Delay Function, distributed projects run at twice the speed of double-checked calculations.  

## Build

genefer22 is under development...  
Any number of the form *b*<sup>2<sup>*n*</sup></sup> + 1 such that 2 &le; *b* < 2,000,000,000 and 14 &le; *n* &le; 22 can be tested on GPU.  
This version is compiled with gcc and was tested on Windows and Linux (Ubuntu).  

## TODO

- save & restore context
- Boinc interface
