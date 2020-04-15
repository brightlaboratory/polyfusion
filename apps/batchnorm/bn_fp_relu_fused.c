#ifndef SH
#define SH 1
#endif // !SH

#ifndef SW
#define SW 1
#endif // !SW

void bn_fp_relu_fused(
	const int nImg, const int nFm, const int ifh, const int ifw,
	const int ofh, const int ofw,
	float input[nImg][nFm][ifh][ifw],
	float input_add[nImg][nFm][ifh][ifw], float output[nImg][nFm][ofh][ofw],
	int norm_type, float expectval[nFm], float rcpstddev[nFm], float variance[nFm],
	float beta[nFm], float gamma[nFm])
{
	const float nhw = (float)(nImg * ifh * ifw);
	const float recp_nhw = 1.0f / nhw;
	const float sqrt_eps = 1e-7f;

	int img, fm, hi, wi, ho, wo;

	if (norm_type == 0) {
		printf("norm_type = 0 is not supported. Exiting\n");
		exit(1);
	}


#pragma scop
#pragma omp parallel for private(img, fm, hi, wi, ho, wo)
	for (img = 0; img < nImg; img++) {
		for (fm = 0; fm < nFm; fm++) {
			for (hi = 0, ho = 0; hi < ifh; hi += SH, ho++) {
				for (wi = 0, wo = 0; wi < ifw; wi += SW, wo++) {
					/* BN + scale: gamma, shift: beta */
					output[img][fm][ho][wo] =
						gamma[fm] *
						(input[img][fm][hi][wi] - expectval[fm]) * rcpstddev[fm]
						+ beta[fm];

//					output[img][fm][ho][wo] += input_add[img][fm][hi][wi];
					output[img][fm][ho][wo] = (output[img][fm][ho][wo] < 0.0) ? 0.0f
						: output[img][fm][ho][wo];
				}
			}
		}
	}
#pragma endscop

}
