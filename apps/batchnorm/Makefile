CC=icc
CFLAGS=-O3 -I . -I../libxsmm/include -qopenmp -fno-alias -xCOMMON-AVX512 -mkl=sequential
LDFLAGS=-L../libxsmm/lib -lxsmm -lxsmmext -lpthread
MACROFLAGS=

default: bn_relu

../libxsmm/include/libxsmm.h:
#	rm -rf ../libxsmm/
#	git clone --branch 1.14 https://github.com/hfp/libxsmm.git ../libxsmm
	$(MAKE) realclean -C ../libxsmm
	$(MAKE) AVX=3 BLAS=0 CXX=icc CC=icc FC=ifort -C ../libxsmm

bn_relu: bn_relu.c ../libxsmm/include/libxsmm.h
	$(CC) $(CFLAGS) $(MACROFLAGS) bn_relu.c $(LDFLAGS) -o bn_relu

clean: 
	rm -rf bn_relu
