# Compiler: GCC 11
CC = g++ -m64 -std=c++17
CFLAGS = -Wall -Wextra -fexceptions -ffinite-math-only -fprefetch-loop-arrays -frename-registers
FLAGS_CPU = -O3 -fopenmp
FLAGS_GPU = -g -DGPU

BIN_DIR = bin
SRC_DIR = src

OCL_INC = -I Khronos
OCL_LIB = C:/Windows/System32/OpenCL.dll

OBJS_CPU = main.o transform_sse2.o transform_sse4.o transform_avx.o transform_fma.o transform_512.o
OBJS_GPU = maing.o transform_ocl.o
DEPS_MAIN = $(SRC_DIR)/genefer.h $(SRC_DIR)/transform.h $(SRC_DIR)/gint.h $(SRC_DIR)/timer.h $(SRC_DIR)/pio.h  $(SRC_DIR)/boinc.h
DEPS_TRANSFORM_CPU = $(SRC_DIR)/transformCPU.h $(SRC_DIR)/transform.h $(SRC_DIR)/gint.h $(SRC_DIR)/fp16_80.h
DEPS_TRANSFORM_GPU = $(SRC_DIR)/transformGPU.h $(SRC_DIR)/transform.h $(SRC_DIR)/gint.h $(SRC_DIR)/ocl.h

EXEC_CPU = $(BIN_DIR)/genefer22.exe
EXEC_GPU = $(BIN_DIR)/genefer22g.exe

build: $(EXEC_CPU) $(EXEC_GPU)

run: $(EXEC_GPU)
	./$(EXEC_GPU)

clean:

main.o : $(SRC_DIR)/main.cpp $(DEPS_MAIN)
	$(CC) $(CFLAGS) $(FLAGS_CPU) -c $< -o $@

maing.o : $(SRC_DIR)/main.cpp $(DEPS_MAIN)
	$(CC) $(CFLAGS) $(FLAGS_GPU) $(OCL_INC) -c $< -o $@

transform_sse2.o : $(SRC_DIR)/transform_sse2.cpp $(DEPS_TRANSFORM_CPU)
	$(CC) $(CFLAGS) $(FLAGS_CPU) -mtune=core2 -c $< -o $@
#	$(CC) $(CFLAGS) $(FLAGS_CPU) -mtune=core2 -c -S $< -o sse2.asm

transform_sse4.o : $(SRC_DIR)/transform_sse4.cpp $(DEPS_TRANSFORM_CPU)
	$(CC) $(CFLAGS) $(FLAGS_CPU) -msse4.1 -mtune=westmere -c $< -o $@
#	$(CC) $(CFLAGS) $(FLAGS_CPU) -msse4.1 -mtune=westmere -c -S $< -o sse4.asm

transform_avx.o : $(SRC_DIR)/transform_avx.cpp $(DEPS_TRANSFORM_CPU)
	$(CC) $(CFLAGS) $(FLAGS_CPU) -mavx -mtune=ivybridge -c $< -o $@
#	$(CC) $(CFLAGS) $(FLAGS_CPU) -mavx -mtune=ivybridge -c -S $< -o avx.asm

transform_fma.o : $(SRC_DIR)/transform_fma.cpp $(DEPS_TRANSFORM_CPU)
	$(CC) $(CFLAGS) $(FLAGS_CPU) -mavx -mfma -c $< -o $@
#	$(CC) $(CFLAGS) $(FLAGS_CPU) -mavx -mfma -c -S $< -o fma.asm

transform_512.o : $(SRC_DIR)/transform_512.cpp $(DEPS_TRANSFORM_CPU)
	$(CC) $(CFLAGS) $(FLAGS_CPU) -mavx512f -c $< -o $@
#	$(CC) $(CFLAGS) $(FLAGS_CPU) -mavx512f -c -S $< -o 512.asm

transform_ocl.o : $(SRC_DIR)/transform_ocl.cpp $(DEPS_TRANSFORM_GPU)
	$(CC) $(CFLAGS) $(FLAGS_GPU) $(OCL_INC) -c $< -o $@

$(EXEC_CPU): $(OBJS_CPU)
	$(CC) $(FLAGS_CPU) -static $^ -lgmp -o $@

$(EXEC_GPU): $(OBJS_GPU)
	$(CC) $(FLAGS_GPU) -static $^ $(OCL_LIB) -lgmp -o $@
