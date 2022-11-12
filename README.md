# genefer 22
Generalized Fermat Prime search program

## About

**genefer22** is an [OpenMP®](https://www.openmp.org/) application on CPU and an [OpenCL™](https://www.khronos.org/opencl/) application on GPU.  
It performs a fast probable primality test for numbers of the form *b*<sup>2<sup>*n*</sup></sup> + 1 with [Fermat test](https://en.wikipedia.org/wiki/Fermat_primality_test).  
Yves Gallot implemented a new test based on right-angle convolution in [Proth.exe](https://www.ams.org/journals/mcom/2002-71-238/S0025-5718-01-01350-3/S0025-5718-01-01350-3.pdf) in 1999. [genefer](https://doi.org/10.5334/jors.ca) was a free source code created in 2001. In 2009, Mark Rodenkirch and David Underbakke wrote an implementation for x64 and in 2010, Shoichiro Yamada and Ken Brazier for CUDA. In 2011, Michael Goetz, Ronald Schneider and Iain Bethune added Boinc API and since then *genefer* has been extensively used by [PrimeGrid](https://www.primegrid.com/forum_forum.php?id=75) computing project. In 2013, Yves Gallot wrote some implementations for OpenCL using Number Theoretic Transforms. In 2014, a z-Transform replaced the original weighted transform.  

*genefer22* is a new C++ application, created in 2022. *genefer* was originally written in C and a new design was necessary to achieve new levels of complexity.  
It implements an [Efficient Modular Exponentiation Proof Scheme](https://arxiv.org/abs/2209.15623) discovered by Darren Li.
The test is validated with [Gerbicz - Li](https://www.mersenneforum.org/showthread.php?t=22510) error checking and a proof is generated with ([Pietrzak - Li](https://eprint.iacr.org/2018/627.pdf)) algorithm. Thanks to the Verifiable Delay Function, distributed projects run at twice the speed of double-checked calculations.  

Any number of the form *b*<sup>2<sup>*n*</sup></sup> + 1 such that 2 &le; *b* < 2,000,000,000 and 12 &le; *n* &le; 23 can be tested on GPU.  
On CPU, the limit is fuzzy and is *b* ~ 500M (*n* = 12), 380M (*n* = 13), 300M (*n* = 14), 230M (*n* = 15), 170M (*n* = 16), 130M (*n* = 17), 95M (*n* = 18), 70M (*n* = 19), 55M (*n* = 20), 45M (*n* = 21), 35M (*n* = 22), 25M (*n* = 23).  

## Build

Releases are available as tags.  
 - 22.11.3

They are compiled using:  
 - genefer22: Ubuntu 22.04 amd64, gcc 11.3  
 - genefer22g: Ubuntu 18.04 amd64, gcc 7.5  
 - genefer22_x86, genefer22g_x86: Lubuntu 18.04 i386, gcc 7.5  
 - genefer22.exe, genefer22g.exe, genefer22_x86.exe, genefer22g_x86: Windows - MSYS2, gcc 12.2  
 - genefer22_macIntel, genefer22g_macIntel: MacOS 10.13, llvm/clang 15  
 - genefer22_macARM, genefer22g_macARM: MacOS, llvm/clang 15  
 - genefer22_arm64: Ubuntu 22.04 arm64, gcc 11.2  

They are tested on:  
 - Ubuntu 22.04 amd64  
 - Windows 11  
 - Windows 10 64-bit  
 - MacOS Apple M1  
 - MacOS Intel Core  
 - Debian 11 amd64  
 - Ubuntu 16.04 amd64  
 - Ubuntu 16.04 i386  
 - Ubuntu 22.04 arm64  

## TODO

 - add FP64 transform on GPU (for ratio FP64 >= 1/4 INT32).
