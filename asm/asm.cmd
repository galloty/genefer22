ptxas -o tmp.elf -O3 -m64 -arch sm_120 -c pgm.ptx
nvdisasm -c tmp.elf > genefer_50.asm
ptxas -o tmp.elf -O3 -m64 -arch sm_89 -c pgm.ptx
nvdisasm -c tmp.elf > genefer_40.asm
ptxas -o tmp.elf -O3 -m64 -arch sm_86 -c pgm.ptx
nvdisasm -c tmp.elf > genefer_30.asm
ptxas -o tmp.elf -O3 -m64 -arch sm_75 -c pgm.ptx
nvdisasm -c tmp.elf > genefer_20.asm
ptxas -o tmp.elf -O3 -m64 -arch sm_61 -c pgm.ptx
nvdisasm -c tmp.elf > genefer_10.asm
del tmp.elf
