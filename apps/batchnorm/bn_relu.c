// Some of the code here is borrowed from the LIBXSMM library: https://github.com/hfp/libxsmm/
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <string.h>

#define USE_LIBXSMM

#if defined(USE_LIBXSMM)
#include <libxsmm.h>
/* function-pointer to LIBXSMM kernel */
libxsmm_smmfunction fwd_gemm;
#endif

#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))

#define NUM_TRIALS 3

#ifndef SH
#define SH 1
#endif // !SH

#ifndef SW
#define SW 1
#endif // !SW

#ifndef GEMM_BLOCK
#define GEMM_BLOCK 64
#endif // !GEMM_BLOCK

#include "naive_bn_fp_relu.c"
#include "bn_fp_relu_fused.c"

typedef struct {
	double max_rel_err;
	double max_abs_err;
	double l2_rel_err;
	double one_norm_ref;
	double one_norm_test;
} correctness_t;

void zero_buf(float* buf, long size) {
	int i;
	for (i = 0; i < size; ++i) {
		buf[i] = 0.0f;
	}
}


void init_buf(float* buf, long size, int initPos, int initOne)
{
	int i;
	zero_buf(buf, size);
	for (i = 0; i < size; ++i) {
		buf[i] = (float)((initOne != 0) ? 1.0 : drand48());
	}
}

void compare_buf(float* ref, float* test, long size, correctness_t* norms)
{
	int i;
	double diff, rel_err;

	norms->max_rel_err = 0.;
	norms->max_abs_err = 0.;
	norms->l2_rel_err = 0.;
	norms->one_norm_ref = 0.;
	norms->one_norm_test = 0.;

	for (i = 0; i < size; ++i) {
		norms->one_norm_ref += (double)ref[i];
		norms->one_norm_test += (double)test[i];
		diff = fabs((double)ref[i] - (double)test[i]);
		norms->l2_rel_err += (diff*diff);
		rel_err = 0.0;
		if (diff > 0.0) {
			rel_err = diff / fabs((double)ref[i]);
		}
		if (rel_err > norms->max_rel_err) {
			norms->max_rel_err = rel_err;
#if 0
			printf("MISMATCH@ %3d: A=%12.8g  B=%12.8g (E:%12.4e) (R:%12.4e)\n", i, ref[i], test[i], diff, rel_err);
#endif
		}
		if (diff > norms->max_abs_err) {
			norms->max_abs_err = diff;
		}
#if 0
		if (diff > 1.0) {
			printf("MISMATCH@ %3d: A=%12.8g  B=%12.8g (E:%12.4e)\n", i, ref[i], test[i], diff);
		}
#endif

	}
	norms->l2_rel_err = sqrt(norms->l2_rel_err);
}



int main(int argc, char **argv) {
	correctness_t norms_fwd;
	memset(&norms_fwd, 0, sizeof(norms_fwd));

	/* some parameters we can overwrite via cli,
	   default is some inner layer of overfeat */
	int iters = 1;         /* repetitions of benchmark */
	int nImg = 32;          /* mini-batch size, "N" */
	int ifw = 14;           /* input width, "W" */
	int ifh = 18;           /* input height, "H" */
	int nFm = 256;         /* number of feature maps */
	int version = 1;
	int check_correctness = 1;

	unsigned long long l_start, l_end;
	double l_total = 0.0;
	double flops = 0.0;

	/* reading new values from cli */
	int i = 1;
	if (argc > i) iters = atoi(argv[i++]);
	if (argc > i) nImg = atoi(argv[i++]);
	if (argc > i) ifw = atoi(argv[i++]);
	if (argc > i) ifh = atoi(argv[i++]);
	if (argc > i) nFm = atoi(argv[i++]);
	if (argc > i) version = atoi(argv[i++]);
	if (argc > i) check_correctness = atoi(argv[i++]);

	printf("version = %d\n", version);

	/* apply stride in both dimensions */
	const int ofh = ifh / SH;
	const int ofw = ifw / SW;


	/* print some summary */
	printf("##########################################\n");
	printf("#                Setting Up              #\n");
	printf("##########################################\n");
	printf("PARAMS: iters:%d  nImg:%d  ifw:%d  ifh:%d  nFm:%d  version:%d  check_correctness:%d  ofh:%d  ofw:%d  \n",
		iters, nImg, ifw, ifh, nFm, version, check_correctness, ofh, ofw);


	printf("SIZE Input  (MB): %10.2f MiB\n", (double)(nImg*nFm*ifh*ifw * sizeof(float)) / (1024.0*1024.0));
	printf("SIZE Output (MB): %10.2f MiB\n", (double)(nImg*nFm*ofh*ofw * sizeof(float)) / (1024.0*1024.0));
	printf("SIZE Input   (1): %10.2f MiB\n", (double)(1 * nFm*ifh*ifw * sizeof(float)) / (1024.0*1024.0));
	printf("SIZE Output  (1): %10.2f MiB\n", (double)(1 * nFm*ofh*ofw * sizeof(float)) / (1024.0*1024.0));

	printf("Allocating data\n");
	/* allocate data */
	float(*input)[nFm][ifh][ifw] =
		(float*)libxsmm_aligned_malloc(nImg*nFm*ifh*ifw * sizeof(float), 2097152);

	float(*input_add)[nFm][ifh][ifw] =
		(float*)libxsmm_aligned_malloc(nImg*nFm*ifh*ifw * sizeof(float), 2097152);

	float(*output)[nFm][ofh][ofw] =
		(float*)libxsmm_aligned_malloc(nImg*nFm*ofh*ofw * sizeof(float), 2097152);

	float(*check_output)[nFm][ofh][ofw] =
		(float*)libxsmm_aligned_malloc(nImg*nFm*ofh*ofw * sizeof(float), 2097152);

	float *expectval = (float*)libxsmm_aligned_malloc(nFm * sizeof(float), 2097152);
	float *rcpstddev = (float*)libxsmm_aligned_malloc(nFm * sizeof(float), 2097152);
	float *variance = (float*)libxsmm_aligned_malloc(nFm * sizeof(float), 2097152);
	float *beta = (float*)libxsmm_aligned_malloc(nFm * sizeof(float), 2097152);
	float *gamma = (float*)libxsmm_aligned_malloc(nFm * sizeof(float), 2097152);


	printf("Initializing data\n");
	/* initialize data */
	srand48(1);
	init_buf(&input[0][0][0][0], nImg*nFm*ifh*ifw, 0, 0);
	init_buf(&input_add[0][0][0][0], nImg*nFm*ifh*ifw, 0, 0);
	zero_buf(&output[0][0][0][0], nImg*nFm*ofh*ofw);
	zero_buf(&check_output[0][0][0][0], nImg*nFm*ofh*ofw);
	zero_buf(&expectval[0], nFm);
	zero_buf(&rcpstddev[0], nFm);
	zero_buf(&variance[0], nFm);
	init_buf(&gamma[0], nFm, 0, 1);
	zero_buf(&beta[0], nFm);
	init_buf(&gamma[0], nFm, 0, 0);
	init_buf(&beta[0], nFm, 0, 0);

	flops = (double)nImg * (double)nFm * (double)nFm * (double)ofh * (double)ofw * 4.0 * (double)iters;

	if (check_correctness) {
		printf("##########################################\n");
		printf("#   Correctness - FWD (custom-Storage)   #\n");
		printf("##########################################\n");
		printf("Calling naive_bn_fp_relu\n");

		naive_bn_fp_relu(
			nImg, nFm, ifh, ifw,
			ofh, ofw,
			input,
			input_add, output,
			0, expectval, rcpstddev, variance,
			beta, gamma);

		l_start = libxsmm_timer_tick();
		bn_fp_relu_fused(
			nImg, nFm, ifh, ifw,
			ofh, ofw,
			input,
			input_add, check_output,
			1, expectval, rcpstddev, variance,
			beta, gamma);
		l_end = libxsmm_timer_tick();
		l_total = libxsmm_timer_duration(l_start, l_end);

		printf("input: \n");
		printf("%f %f %f\n", input[0][0][0][0],
			input[nImg / 2][nFm / 2][ifh / 2][ifw / 2],
			input[nImg - 1][nFm - 1][ifh - 1][ifw - 1]);

		printf("input_add: \n");
		printf("%f %f %f\n", input_add[0][0][0][0],
			input_add[nImg / 2][nFm / 2][ifh / 2][ifw / 2],
			input_add[nImg - 1][nFm - 1][ifh - 1][ifw - 1]);

		printf("expectval: \n");
		printf("%f %f %f\n", expectval[0], expectval[nFm / 2], expectval[nFm - 1]);

		printf("rcpstddev: \n");
		printf("%f %f %f\n", rcpstddev[0], rcpstddev[nFm / 2], rcpstddev[nFm - 1]);

		printf("variance: \n");
		printf("%f %f %f\n", variance[0], variance[nFm / 2], variance[nFm - 1]);

		printf("beta: \n");
		printf("%f %f %f\n", beta[0], beta[nFm / 2], beta[nFm - 1]);

		printf("gamma: \n");
		printf("%f %f %f\n", gamma[0], gamma[nFm / 2], gamma[nFm - 1]);

		printf("output: \n");
		printf("%f %f %f\n", output[0][0][0][0],
			output[nImg / 2][nFm / 2][ofh / 2][ofw / 2],
			output[nImg - 1][nFm - 1][ofh - 1][ofw - 1]);

		printf("check_output: \n");
		printf("%f %f %f\n", check_output[0][0][0][0],
			check_output[nImg / 2][nFm / 2][ofh / 2][ofw / 2],
			check_output[nImg - 1][nFm - 1][ofh - 1][ofw - 1]);

		// compare
		compare_buf(output, check_output, nImg*nFm*ofh*ofw, &norms_fwd);
		printf("             1-norm of reference: %f\n", norms_fwd.one_norm_ref);
		printf("             1-norm of GEMM-code: %f\n", norms_fwd.one_norm_test);
		printf("      L2-error-norm of GEMM-code: %f\n", norms_fwd.l2_rel_err);
		printf("    inf-norm of comp. rel. error: %f\n", norms_fwd.max_rel_err);
		printf("    inf-norm of comp. abs. error: %f\n", norms_fwd.max_abs_err);

	}
	else {
		/* Warm up */
		bn_fp_relu_fused(
			nImg, nFm, ifh, ifw,
			ofh, ofw,
			input,
			input_add, check_output,
			1, expectval, rcpstddev, variance,
			beta, gamma);
	}

	printf("##########################################\n");
	printf("#   Performance - FWD (custom-Storage)   #\n");
	printf("##########################################\n");


	int trial;
	double min_l_total = 0.0;
	for (trial = 0; trial < NUM_TRIALS; trial++) {

		if (trial == 0) {
			min_l_total = l_total;
		}
		else {
			min_l_total = min(min_l_total, l_total);
		}
	}

	l_total = min_l_total;

	printf("Elapsed time of padded_conv_fp = %f seconds\n", l_total);
	printf("GFLOP  = %.5g\n", flops*1e-9 / (double)iters);
	printf("fp time = %.5g\n", ((double)(l_total / iters)));
	printf("Real_GFLOPS =%.5g\n", (flops*1e-9) / l_total);


	libxsmm_free(input);
	libxsmm_free(input_add);
	libxsmm_free(output);
	libxsmm_free(expectval);
	libxsmm_free(rcpstddev);
	libxsmm_free(variance);
	libxsmm_free(beta);
	libxsmm_free(gamma);

	return 0;
}
