#include "padded_conv_relu_fp_libxsmm_core_gemm.c"

double padded_conv2d_relu_fp(
	int nImg, int nIfm, int nOfm, int ifhp, int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float input[nImg][nIfm / GEMM_BLOCK][ifhp][ifwp][GEMM_BLOCK], float output[nImg][nOfm / GEMM_BLOCK][ofhp][ofwp][GEMM_BLOCK], const float filter[nOfm / GEMM_BLOCK][nIfm / GEMM_BLOCK][kh][kw][GEMM_BLOCK][GEMM_BLOCK], int version, int iters,
	const float naive_input[nImg][nIfm][ifhp][ifwp], const float naive_filter[nOfm][nIfm][kh][kw],
	float check_output[nImg][nOfm][ofhp][ofwp])
{
	int copyGEMMOutputToNCHWformat = 1;
	unsigned long long l_start, l_end;
	double l_total = 0.0;
	int i;
	/* declare a physical padded buffer */


	float(*pad_gemm_input)[nIfm / GEMM_BLOCK][ifhp + 2 * pad_h][ifwp + 2 * pad_w][GEMM_BLOCK] = (float*)libxsmm_aligned_malloc(nImg*nIfm*(ifhp + 2 * pad_h)*(ifwp + 2 * pad_w) * sizeof(float), 2097152);
	zero_buf(&pad_gemm_input[0][0][0][0][0], (nImg)*(nIfm / GEMM_BLOCK)*(ifhp + 2 * pad_h)*(ifwp + 2 * pad_w) * GEMM_BLOCK);
	copy_GEMM_to_PADDED_GEMM(nImg, ifhp, ifwp, nIfm, pad_h, pad_w, input, pad_gemm_input);


	if (version == 2) {
		l_start = libxsmm_timer_tick();
		for (i = 0; i < iters; i++) {
			padded_conv_relu_fp_libxsmm_core_gemm(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
				ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
				pad_w_out, kh, kw, stride_h, stride_w, pad_gemm_input, output, filter, iters);
		}

		l_end = libxsmm_timer_tick();
	}
	else if (version == 3) {
		l_start = libxsmm_timer_tick();
		for (i = 0; i < iters; i++) {
			padded_conv_relu_fp_libxsmm_core_gemm_fn(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
				ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
				pad_w_out, kh, kw, stride_h, stride_w, pad_gemm_input, output, filter, iters);
		}

		l_end = libxsmm_timer_tick();
	}
	else if (version == 4) {
		l_start = libxsmm_timer_tick();
		for (i = 0; i < iters; i++) {
			padded_conv_relu_fp_libxsmm_core_gemm_fn_fused(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
				ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
				pad_w_out, kh, kw, stride_h, stride_w, pad_gemm_input, output, filter, iters);
		}

		l_end = libxsmm_timer_tick();
	}
	else if (version == 5) {
		l_start = libxsmm_timer_tick();
		for (i = 0; i < iters; i++) {
			padded_conv_relu_fp_libxsmm_core_gemm_fused(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
				ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
				pad_w_out, kh, kw, stride_h, stride_w, pad_gemm_input, output, filter, iters);
		}

		l_end = libxsmm_timer_tick();
	}
	else if (version == 6) {
		l_start = libxsmm_timer_tick();
		for (i = 0; i < iters; i++) {
			padded_conv_relu_fp_libxsmm_core_gemm_fused2(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
				ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
				pad_w_out, kh, kw, stride_h, stride_w, pad_gemm_input, output, filter, iters);
		}

		l_end = libxsmm_timer_tick();
	}
	else if (version == 7) {
		l_start = libxsmm_timer_tick();
		for (i = 0; i < iters; i++) {
			padded_conv_relu_fp_libxsmm_core_gemm_fused2(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
				ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
				pad_w_out, kh, kw, stride_h, stride_w, pad_gemm_input, output, filter, iters);
		}

		l_end = libxsmm_timer_tick();
	}
	else {
		printf("Incorrect version\n");
		libxsmm_free(pad_gemm_input);
		exit(0);
	}

	if (copyGEMMOutputToNCHWformat) {
		printf("Calling copy_GEMM_to_NCHW\n");
		copy_GEMM_to_NCHW(nImg, ofhp, ofwp, nOfm, output, check_output);
	}

	libxsmm_free(pad_gemm_input);
	l_total = libxsmm_timer_duration(l_start, l_end);
	return l_total;
}