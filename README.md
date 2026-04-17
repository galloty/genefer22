# genefer
Generalized Fermat Prime search program

<!---
[![Linux build](https://github.com/galloty/genefer22/actions/workflows/linux.yml/badge.svg?branch=main)](https://github.com/galloty/genefer22/actions/workflows/linux.yml)
[![macOS build](https://github.com/galloty/genefer22/actions/workflows/macOS.yml/badge.svg?branch=main)](https://github.com/galloty/genefer22/actions/workflows/macOS.yml)
-->
## About

**genefer** is a multithreaded application on CPU and an [OpenCL™](https://www.khronos.org/opencl/) application on GPU.  
It performs a fast probable primality test for numbers of the form *b*<sup>2<sup>*n*</sup></sup> + 1 with [Fermat test](https://en.wikipedia.org/wiki/Fermat_primality_test).  
A slower [deterministic primality test](https://pubs.ams.org/journals/mcom/1975-29-130/S0025-5718-1975-0384673-1) is also available.  

Yves Gallot implemented a new test based on right-angle convolution in [Proth.exe](https://www.ams.org/journals/mcom/2002-71-238/S0025-5718-01-01350-3) in 1999. [genefer](https://doi.org/10.5334/jors.ca) was a free source code created in 2001. In 2009, Mark Rodenkirch and David Underbakke wrote an implementation for x64 and in 2010, Shoichiro Yamada and Ken Brazier for CUDA. In 2011, Michael Goetz, Ronald Schneider and Iain Bethune added Boinc API and since then *genefer* has been extensively used by [PrimeGrid](https://www.primegrid.com/forum_forum.php?id=75) computing project. In 2013, Yves Gallot wrote different implementations for OpenCL using Number Theoretic Transforms. In 2014, a z-Transform replaced the original weighted transform.  

*genefer* version 22+ is a new C++ application, created in 2022. The previous versions inherited the originally code written in C and a new design was necessary to achieve new levels of complexity.  

It implements an [Efficient Modular Exponentiation Proof Scheme](https://arxiv.org/abs/2209.15623) discovered by Darren Li.
The test is validated with [Gerbicz - Li](https://www.mersenneforum.org/showthread.php?t=22510) error checking and a proof is generated with ([Pietrzak - Li](https://eprint.iacr.org/2018/627.pdf)) algorithm. Thanks to the Verifiable Delay Function, distributed projects run at twice the speed of double-checked calculations.  

Any number of the form *b*<sup>2<sup>*n*</sup></sup> + 1 such that 1024 &le; *b* < 2,000,000,000 and 12 &le; *n* &le; 23 can be tested on GPU.  
On CPU, the code is optimized for PrimeGrid tests and the current limits are *b* = 2000M (*n* = 12, 13, 14, 15), *b* ~ 1500M (*n* = 16), 1000M (*n* = 17, 18), 70M (*n* = 19), 55M (*n* = 20), 45M (*n* = 21), 35M (*n* = 22), 25M (*n* = 23).  

## Operating system

 - Linux x64 and arm64  
 - Windows x64 and arm64  
 - MacOS x64 and arm64  
 - Android arm64  

The 32-bit versions are no longer supported.  

## Build

Select the [makefile](https://github.com/galloty/genefer22/tree/main/genefer) of your target. On Windows, [MSYS2](https://www.msys2.org/) distribution and building platform can be installed.  
The compiler (gcc or clang) can be changed, Boinc interface is optional.  
The default settings are: gcc on x64 and clang on arm64, linked to Boinc.  

Binaries are validated using:  
 - genefer: Ubuntu 24.04, gcc 13.3 and clang 20.1
 - geneferg: Ubuntu 18.04, gcc 7.5  
 - genefer.exe, geneferg.exe: Windows 11 - MSYS2, gcc 15.2 and clang 22.1  
 - genefer_arm, geneferg_arm: Ubuntu 22.04, gcc 11.4 and clang 14.0
 - genefer_arm.exe, geneferg_arm.exe: Windows 11 on Arm - MSYS2, ?  
 - genefer_android_arm: Android 12, NDK r29 - clang 21.0  
 - genefer_mac_x64, geneferg_mac_x64: MacOS 10.13, clang 15  
 - genefer_mac_arm, geneferg_mac_arm: MacOS 12, clang 15  

## Licence

**genefer** is free source code, under the MIT license (see [LICENSE](https://github.com/galloty/genefer22/blob/main/LICENSE)). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
