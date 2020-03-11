
#include "naive_bn_fp_relu.c"
#include "bn_fp_relu_fused.c"

double bn_highperf(
	const int nImg, const int nFm, const int ifh, const int ifw,
	const int ofh, const int ofw,
	float input[nImg][nFm][ifh][ifw],
	float input_add[nImg][nFm][ifh][ifw], float output[nImg][nFm][ofh][ofw],
	int norm_type, float expectval[nFm], float rcpstddev[nFm], float variance[nFm],
	float beta[nFm], float gamma[nFm], int version, int iters) {

	unsigned long long l_start, l_end;
	double l_total = 0.0;

	int i;

	if (version == 0) {

		l_start = libxsmm_timer_tick();

		for (i = 0; i < iters; i++) {
			naive_bn_fp_relu(
				nImg, nFm, ifh, ifw,
				ofh, ofw,
				input,
				input_add, output,
				1, expectval, rcpstddev, variance,
				beta, gamma);
		}

		l_end = libxsmm_timer_tick();
		l_total = libxsmm_timer_duration(l_start, l_end);

	}
	else if (version == 1) {

		l_start = libxsmm_timer_tick();

		for (i = 0; i < iters; i++) {
			bn_fp_relu_fused(
				nImg, nFm, ifh, ifw,
				ofh, ofw,
				input,
				input_add, output,
				1, expectval, rcpstddev, variance,
				beta, gamma);
		}

		l_end = libxsmm_timer_tick();
		l_total = libxsmm_timer_duration(l_start, l_end);

	}
	else {
		printf("Version %d not supported. Exiting\n", version);
		exit(1);
	}

	return l_total;
}