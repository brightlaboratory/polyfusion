#ifndef SH
#define SH 1
#endif // !SH

#ifndef SW
#define SW 1
#endif // !SW

void naive_bn_fp_relu_fn(
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
#pragma omp parallel for private(img, hi, wi)
		for (fm = 0; fm < nFm; fm++) {
			float ch_sum = 0.0f;
			float ch_sumsq = 0.0f;
			float tbmean = 0.0f;
			float tbmeansq = 0.0f;
			float tsqbmean = 0.0f;
			float tbrstd = 0.0f;
			float tvariance = 0.0f;

			for (img = 0; img < nImg; img++) {
				for (hi = 0; hi < ifh; hi++) {
					for (wi = 0; wi < ifw; wi++) {
						const float input_val = input[img][fm][hi][wi];
						ch_sum += input_val;
						ch_sumsq += (input_val * input_val);
					}
				}
			}

			tbmean = recp_nhw * ch_sum;
			tbmeansq = tbmean * tbmean;
			tsqbmean = recp_nhw * ch_sumsq;
			tvariance = tsqbmean - tbmeansq;
			tbrstd = (float)(1.0 / sqrt(tvariance + sqrt_eps));
			expectval[fm] = tbmean;
			rcpstddev[fm] = tbrstd;
			variance[fm] = tvariance;
		}
	}


#pragma scop
#pragma omp parallel for private(img, fm, hi, wi, ho, wo)
	for (img = 0; img < nImg; img++) {
		for (fm = 0; fm < nFm; fm++) {
			for (hi = 0, ho = 0; hi < ifh; hi += SH, ho++) {
				for (wi = 0, wo = 0; wi < ifw; wi += SW, wo++) {
					/* BN + scale (gamma, beta) */
					output[img][fm][ho][wo] =
						gamma[fm] *
						(input[img][fm][hi][wi] - expectval[fm]) * rcpstddev[fm]
						+ beta[fm];
				}
			}
		}
	}

#pragma endscop

}
