# genefer
Generalized Fermat Prime search program

<!---
[![Linux build](https://github.com/galloty/genefer22/actions/workflows/linux.yml/badge.svg?branch=main)](https://github.com/galloty/genefer22/actions/workflows/linux.yml)
[![macOS build](https://github.com/galloty/genefer22/actions/workflows/macOS.yml/badge.svg?branch=main)](https://github.com/galloty/genefer22/actions/workflows/macOS.yml)
-->
## About

**genefer** is an [OpenMP®](https://www.openmp.org/) application on CPU and an [OpenCL™](https://www.khronos.org/opencl/) application on GPU.  
It performs a fast probable primality test for numbers of the form *b*<sup>2<sup>*n*</sup></sup> + 1 with [Fermat test](https://en.wikipedia.org/wiki/Fermat_primality_test).  
Yves Gallot implemented a new test based on right-angle convolution in [Proth.exe](https://www.ams.org/journals/mcom/2002-71-238/S0025-5718-01-01350-3/S0025-5718-01-01350-3.pdf) in 1999. [genefer](https://doi.org/10.5334/jors.ca) was a free source code created in 2001. In 2009, Mark Rodenkirch and David Underbakke wrote an implementation for x64 and in 2010, Shoichiro Yamada and Ken Brazier for CUDA. In 2011, Michael Goetz, Ronald Schneider and Iain Bethune added Boinc API and since then *genefer* has been extensively used by [PrimeGrid](https://www.primegrid.com/forum_forum.php?id=75) computing project. In 2013, Yves Gallot wrote different implementations for OpenCL using Number Theoretic Transforms. In 2014, a z-Transform replaced the original weighted transform.  

*genefer* version 22+ is a new C++ application, created in 2022. It was originally written in C and a new design was necessary to achieve new levels of complexity.  
It implements an [Efficient Modular Exponentiation Proof Scheme](https://arxiv.org/abs/2209.15623) discovered by Darren Li.
The test is validated with [Gerbicz - Li](https://www.mersenneforum.org/showthread.php?t=22510) error checking and a proof is generated with ([Pietrzak - Li](https://eprint.iacr.org/2018/627.pdf)) algorithm. Thanks to the Verifiable Delay Function, distributed projects run at twice the speed of double-checked calculations.  

Any number of the form *b*<sup>2<sup>*n*</sup></sup> + 1 such that 2 &le; *b* < 2,000,000,000 and 12 &le; *n* &le; 23 can be tested on GPU.  
On CPU, the code is optimized for PrimeGrid tests and the limits are *b* = 2000M (*n* = 12, 13, 14, 15), *b* ~ 1500M (*n* = 16), 1000M (*n* = 17), 95M (*n* = 18), 70M (*n* = 19), 55M (*n* = 20), 45M (*n* = 21), 35M (*n* = 22), 25M (*n* = 23).  

## Build

Release 22.12.2, 23.01.0 and 23.07.0 are currently PrimeGrid applications.  

Binaries are generated using:  
 - genefer: Ubuntu 22.04.4 amd64, gcc 11.4  
 - geneferg: Ubuntu 18.04.6 amd64, gcc 7.5  
 - genefer_x86, geneferg_x86: Lubuntu 18.04.6 i386, gcc 7.5  
 - genefer.exe, geneferg.exe, genefer_x86.exe, geneferg_x86.exe: Windows - MSYS2, gcc 13.2  
 - genefer_macIntel, geneferg_macIntel: MacOS 10.13, llvm/clang 15  
 - genefer_macARM, geneferg_macARM: MacOS, llvm/clang 15  
 - genefer_arm64: Ubuntu 22.04 arm64, gcc 11.2  

## TODO

 - add FP64 transform on GPU (if FP64 >= 1/4 x INT32 Units).  
