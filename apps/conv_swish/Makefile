CC=icc
CFLAGS=-O3 -I . -I./libxsmm/include -qopenmp -fno-alias -xCOMMON-AVX512 -mkl=sequential
LDFLAGS=-L./libxsmm/lib -lxsmm -lxsmmext -lpthread
MACROFLAGS=

default: conv2d_swish

./libxsmm/include/libxsmm.h:
	rm -rf libxsmm/
	git clone --branch 1.14 https://github.com/hfp/libxsmm.git
	$(MAKE) realclean -C ./libxsmm
	$(MAKE) AVX=3 BLAS=0 CXX=icc CC=icc FC=ifort -C ./libxsmm 

conv2d_swish: conv2d_swish.c ./libxsmm/include/libxsmm.h
	$(CC) $(CFLAGS) $(MACROFLAGS) conv2d_swish.c $(LDFLAGS) -o conv2d_swish

clean: 
	rm -rf conv2d_swish

