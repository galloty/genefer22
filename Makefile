# Compiler: GCC 11
CC = g++ -m64 -std=c++17
CFLAGS = -Wall -Wextra -fexceptions -ffinite-math-only -fprefetch-loop-arrays -frename-registers
BFLAGS = -O3 -fopenmp

BIN_DIR = bin
SRC_DIR = src

OBJS = main.o transform_sse2.o transform_sse4.o transform_avx.o transform_fma.o transform_512.o
DEPSM = $(SRC_DIR)/genefer.h $(SRC_DIR)/transform.h $(SRC_DIR)/gint.h
DEPST = $(SRC_DIR)/transform.h $(SRC_DIR)/transformCPU.h $(SRC_DIR)/fp16_80.h

EXEC = $(BIN_DIR)/genefer22.exe

build: $(EXEC)

run: $(EXEC)
	./$(EXEC)

clean:

main.o : $(SRC_DIR)/main.cpp $(DEPSM)
	$(CC) $(CFLAGS) $(BFLAGS) -c $< -o $@
#	$(CC) $(CFLAGS) $(BFLAGS) -S -masm=intel $<

transform_sse2.o : $(SRC_DIR)/transform_sse2.cpp $(DEPST)
	$(CC) $(CFLAGS) $(BFLAGS) -mtune=core2 -c $< -o $@
#	$(CC) $(CFLAGS) $(BFLAGS) -mtune=core2 -c -S $< -o sse2.asm

transform_sse4.o : $(SRC_DIR)/transform_sse4.cpp $(DEPST)
	$(CC) $(CFLAGS) $(BFLAGS) -msse4.1 -mtune=westmere -c $< -o $@
#	$(CC) $(CFLAGS) $(BFLAGS) -msse4.1 -mtune=westmere -c -S $< -o sse4.asm

transform_avx.o : $(SRC_DIR)/transform_avx.cpp $(DEPST)
	$(CC) $(CFLAGS) $(BFLAGS) -mavx -mtune=ivybridge -c $< -o $@
#	$(CC) $(CFLAGS) $(BFLAGS) -mavx -mtune=ivybridge -c -S $< -o avx.asm

transform_fma.o : $(SRC_DIR)/transform_fma.cpp $(DEPST)
	$(CC) $(CFLAGS) $(BFLAGS) -mavx -mfma -c $< -o $@
#	$(CC) $(CFLAGS) $(BFLAGS) -mavx -mfma -c -S $< -o fma.asm

transform_512.o : $(SRC_DIR)/transform_512.cpp $(DEPST)
	$(CC) $(CFLAGS) $(BFLAGS) -mavx512f -c $< -o $@
#	$(CC) $(CFLAGS) $(BFLAGS) -mavx512f -c -S $< -o 512.asm

$(EXEC): $(OBJS)
	$(CC) $(BFLAGS) -static $^ -lgmp -o $@
