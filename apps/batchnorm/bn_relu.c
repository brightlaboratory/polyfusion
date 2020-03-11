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
		buf[i] = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? drand48() : (0.05 - drand48() / 10.0)));
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
	int ifhp, ifwp, ofhp, ofwp, ofh, ofw;
	int stride_h, stride_w, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out;
	int version = 2;
	int check_correctness = 1;

	correctness_t norms_fwd;
	memset(&norms_fwd, 0, sizeof(norms_fwd));

	/* some parameters we can overwrite via cli,
	   default is some inner layer of overfeat */
	int iters = 1;         /* repetitions of benchmark */
	int ifw = 14;           /* input width, "W" */
	int ifh = 18;           /* input height, "H" */
	int nImg = 32;          /* mini-batch size, "N" */
	int nIfm = 256;         /* number of input feature maps, "C" */
	int nOfm = 512;         /* number of output feature maps, "K" */
	int kh = 3;             /* filter height, "R" */
	int kw = 3;             /* filter width, "S" */
	int pad = 2;            /* padding in output */
	int stride = 1;         /* stride when accessing inputs */

	pad_w = pad;
	pad_h = pad;

	unsigned long long l_start, l_end;
	double l_total = 0.0;
	double flops = 0.0;

	/* reading new values from cli */
	int i = 1;
	if (argc > i) iters = atoi(argv[i++]);
	if (argc > i) ifw = atoi(argv[i++]);
	if (argc > i) ifh = atoi(argv[i++]);
	if (argc > i) nIfm = atoi(argv[i++]);
	if (argc > i) nOfm = atoi(argv[i++]);
	if (argc > i) kw = atoi(argv[i++]);
	if (argc > i) kh = atoi(argv[i++]);
	if (argc > i) pad_w = atoi(argv[i++]);
	if (argc > i) pad_h = atoi(argv[i++]);
	if (argc > i) stride = atoi(argv[i++]);
	if (argc > i) nImg = atoi(argv[i++]);
	if (argc > i) version = atoi(argv[i++]);
	if (argc > i) check_correctness = atoi(argv[i++]);

	printf("version = %d\n", version);

	/* apply stride in both dimensions */
	stride_w = stride;
	stride_h = stride;

	pad_h_in = 0;
	pad_w_in = 0;
	pad_h_out = 0;
	pad_w_out = 0;

	/* deriving some values image size */
	ofh = (ifh + 2 * pad_h - kh) / stride_h + 1;
	ofw = (ifw + 2 * pad_w - kw) / stride_w + 1;
	ifhp = ifh + 2 * pad_h_in;
	ifwp = ifw + 2 * pad_w_in;
	ofhp = ofh + 2 * pad_h_out;
	ofwp = ofw + 2 * pad_w_out;


	printf("PolyScientist config: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
		ofw, ofh, nIfm, nOfm, kw, kh, pad_w, pad_h, nImg, ifwp, ifhp, ofwp, ofhp, stride_w, stride_h);

	/* some empty lines at the beginning */
	printf("\n\n\n");

	/* print some summary */
	printf("##########################################\n");
	printf("#                Setting Up              #\n");
	printf("##########################################\n");
	printf("PARAMS: W:%d  H:%d  N:%d  C:%d  K:%d  R:%d  S:%d  P:%d  Q:%d  STRIDE:%d\n", ifw, ifh, nImg, nIfm, nOfm, kw, kh, ofh, ofw, stride);
	printf("PARAMS: ITERS:%d", iters);
	printf(" InImg %dx%d Padded (%dx%d)\n", ifh, ifw, ifhp, ifwp);
	printf("OutImg %dx%d Padded (%dx%d)\n", ofh, ofw, ofhp, ofwp);
	printf("SIZE Input  (MB): %10.2f MiB\n", (double)(nImg*nIfm*ifhp*ifwp * sizeof(float)) / (1024.0*1024.0));
	printf("SIZE Output (MB): %10.2f MiB\n", (double)(nImg*nOfm*ofhp*ofwp * sizeof(float)) / (1024.0*1024.0));
	printf("SIZE Input   (1): %10.2f MiB\n", (double)(1 * nIfm*ifhp*ifwp * sizeof(float)) / (1024.0*1024.0));
	printf("SIZE Output  (1): %10.2f MiB\n", (double)(1 * nOfm*ofhp*ofwp * sizeof(float)) / (1024.0*1024.0));
	printf("SIZE Weight     : %10.2f MiB\n", (double)(nIfm*nOfm*kw*kh * sizeof(float)) / (1024.0*1024.0));


	printf("Allocating data\n");
	/* allocate data */
	float(*naive_input)[nIfm][ifhp][ifwp] =
		(float*)libxsmm_aligned_malloc(nImg*nIfm*ifhp*ifwp * sizeof(float), 2097152);

	float(*naive_output)[nOfm][ofhp][ofwp] =
		(float*)libxsmm_aligned_malloc(nImg*nOfm*ofhp*ofwp * sizeof(float), 2097152);

	float(*naive_filter)[nIfm][kh][kw] =
		(float*)libxsmm_aligned_malloc(nOfm*nIfm*kh*kw * sizeof(float), 2097152);

	float(*gemm_input)[nIfm / GEMM_BLOCK][ifhp][ifwp][GEMM_BLOCK] = (float*)libxsmm_aligned_malloc(nImg*nIfm*ifhp*ifwp * sizeof(float), 2097152);
	float(*gemm_output)[nOfm / GEMM_BLOCK][ofhp][ofwp][GEMM_BLOCK] = (float*)libxsmm_aligned_malloc(nImg*nOfm*ofhp*ofwp * sizeof(float), 2097152);
	float(*gemm_filter)[nIfm / GEMM_BLOCK][kh][kw][GEMM_BLOCK][GEMM_BLOCK] = (float*)libxsmm_aligned_malloc(nOfm*nIfm*kh*kw * sizeof(float), 2097152);
	float(*check_output)[nOfm][ofhp][ofwp] = (float*)libxsmm_aligned_malloc(nImg*nOfm*ofhp*ofwp * sizeof(float), 2097152);

	printf("Initializing data\n");
	/* initialize data */
	srand48(1);
	init_buf(&naive_input[0][0][0][0], nImg*nIfm*ifhp*ifwp, 0, 0);
	init_buf(&naive_filter[0][0][0][0], nOfm*nIfm*kh*kw, 0, 0);
	zero_buf(&naive_output[0][0][0][0], nImg*nOfm*ofhp*ofwp);
	zero_buf(&gemm_output[0][0][0][0][0], nImg*nOfm*ofhp*ofwp);
	zero_buf(&check_output[0][0][0][0], nImg*nOfm*ofhp*ofwp);


	clock_t start, end;
	double exec_time;
	flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

	if (check_correctness) {
		printf("##########################################\n");
		printf("#   Correctness - FWD (custom-Storage)   #\n");
		printf("##########################################\n");
		printf("Calling naive_conv_fp_relu_fn\n");

		start = clock();
		l_start = libxsmm_timer_tick();

		l_end = libxsmm_timer_tick();
		l_total = libxsmm_timer_duration(l_start, l_end);
		printf("Naive_GFLOPS =%.5g\n", (flops*1e-9) / l_total / (double)iters);

		end = clock();
		exec_time = (double)(end - start) / CLOCKS_PER_SEC;
		printf("Total time of naive_conv_fp_relu_fn = %f seconds\n", exec_time);

		printf("Printing input values\n");
		printf("%f %f %f\n", naive_input[0][0][0][0], naive_input[nImg / 2][nIfm / 2][ifhp / 2][ifwp / 2], naive_input[nImg - 1][nIfm - 1][ifhp - 1][ifwp - 1]);
		printf("%f %f %f\n", gemm_input[0][0][0][0][0], gemm_input[nImg / 2][(nIfm / 2) / GEMM_BLOCK][ifhp / 2][ifwp / 2][(nIfm / 2) % GEMM_BLOCK], gemm_input[nImg - 1][(nIfm - 1) / GEMM_BLOCK][ifhp - 1][ifwp - 1][(nIfm - 1) % GEMM_BLOCK]);
		printf("Printing weight values\n");
		printf("%f %f %f\n", naive_filter[0][0][0][0], naive_filter[nOfm / 2][nIfm / 2][kh / 2][kw / 2], naive_filter[nOfm - 1][nIfm - 1][kh - 1][kw - 1]);
		printf("%f %f %f\n", gemm_filter[0][0][0][0][0][0], gemm_filter[(nOfm / 2) / GEMM_BLOCK][(nIfm / 2) / GEMM_BLOCK][kh / 2][kw / 2][(nOfm / 2) % GEMM_BLOCK][(nIfm / 2) % GEMM_BLOCK], gemm_filter[(nOfm - 1) / GEMM_BLOCK][(nIfm - 1) / GEMM_BLOCK][kh - 1][kw - 1][(nOfm - 1) % GEMM_BLOCK][(nIfm - 1) % GEMM_BLOCK]);
		printf("Printing output values\n");
		printf("%f %f %f\n", naive_output[0][0][0][0], naive_output[nImg / 2][nOfm / 2][ofhp / 2][ofwp / 2], naive_output[nImg - 1][nOfm - 1][ofhp - 1][ofwp - 1]);

		printf("Printing check_output values\n");
		printf("%f %f %f\n", check_output[0][0][0][0], check_output[nImg / 2][nOfm / 2][ofhp / 2][ofwp / 2], check_output[nImg - 1][nOfm - 1][ofhp - 1][ofwp - 1]);
		printf("Printing gemm_output values\n");
		printf("%f %f %f\n", gemm_output[0][0][0][0][0], gemm_output[nImg / 2][(nOfm / 2) / GEMM_BLOCK][ofhp / 2][ofwp / 2][(nOfm / 2) % GEMM_BLOCK], gemm_output[nImg - 1][(nOfm - 1) / GEMM_BLOCK][ofhp - 1][ofwp - 1][(nOfm - 1) % GEMM_BLOCK]);


		// compare
		compare_buf(naive_output, check_output, nImg*nOfm*ofhp*ofwp, &norms_fwd);
		printf("             1-norm of reference: %f\n", norms_fwd.one_norm_ref);
		printf("             1-norm of GEMM-code: %f\n", norms_fwd.one_norm_test);
		printf("      L2-error-norm of GEMM-code: %f\n", norms_fwd.l2_rel_err);
		printf("    inf-norm of comp. rel. error: %f\n", norms_fwd.max_rel_err);
		printf("    inf-norm of comp. abs. error: %f\n", norms_fwd.max_abs_err);

	}
	else {
		/* Warm up */
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


	libxsmm_free(gemm_input);
	libxsmm_free(gemm_output);
	libxsmm_free(gemm_filter);
	libxsmm_free(check_output);
	return 0;
}
