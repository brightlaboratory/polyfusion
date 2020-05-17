#ifndef STRIDE_H
#define STRIDE_H 1
#endif // !STRIDE_H

#ifndef STRIDE_W
#define STRIDE_W 1
#endif // !STRIDE_W

#ifndef GEMM_BLOCK
#define GEMM_BLOCK 64
#endif // !GEMM_BLOCK

static inline void padded_conv_relu_fp_libxsmm_core_gemm_fn(int nImg, int nIfm, int nOfm, int ifhp, int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float pad_gemm_input[nImg][nIfm / GEMM_BLOCK][ifhp + 2 * pad_h][ifwp + 2 * pad_w][GEMM_BLOCK], float output[nImg][nOfm / GEMM_BLOCK][ofhp][ofwp][GEMM_BLOCK], const float filter[nOfm / GEMM_BLOCK][nIfm / GEMM_BLOCK][kh][kw][GEMM_BLOCK][GEMM_BLOCK], int iters)
{
	/* loop counters */
	int img, ofm_tile, ofm, ifm_tile, ifm, oj, oi, ij, ii, kj, ki, i;

#pragma scop
#pragma omp parallel for private(ofm_tile, ifm_tile, ij, oj, kj, ki, ii, oi, ifm, ofm)
	for (img = 0; img < nImg; ++img) {
		for (ofm_tile = 0; ofm_tile < nOfm / GEMM_BLOCK; ++ofm_tile) {
			for (ifm_tile = 0; ifm_tile < nIfm / GEMM_BLOCK; ++ifm_tile) {
				for (oj = 0; oj < ofh; ++oj) {
					ij = oj * STRIDE_H;
					for (kj = 0; kj < kh; ++kj) {
						for (ki = 0; ki < kw; ++ki) {
							//GEMM
							for (oi = 0; oi < ofw; ++oi) {
								ii = oi * STRIDE_W;
								for (ofm = 0; ofm < GEMM_BLOCK; ++ofm) {
									for (ifm = 0; ifm < GEMM_BLOCK; ++ifm) {
										output[img][ofm_tile][oj][oi][ofm] +=
											filter[ofm_tile][ifm_tile][kj][ki][ifm][ofm] *
											pad_gemm_input[img][ifm_tile][ij + kj][ii + ki][ifm];
									}
								}
							}
						}
					}
				}
			}
		}
	}


#pragma omp parallel for private(ofm_tile, oj, oi, ofm)
	for (img = 0; img < nImg; ++img) {
		for (ofm_tile = 0; ofm_tile < nOfm / GEMM_BLOCK; ++ofm_tile) {
			for (oj = 0; oj < ofh; ++oj) {
				for (oi = 0; oi < ofw; ++oi) {
					for (ofm = 0; ofm < GEMM_BLOCK; ++ofm) {
						output[img][ofm_tile][oj][oi][ofm] =
							(output[img][ofm_tile][oj][oi][ofm] < 0.0f) ? 0.0f :
							output[img][ofm_tile][oj][oi][ofm];
					}
				}
			}
		}
	}
#pragma endscop
}



static inline void padded_conv_relu_fp_libxsmm_core_gemm_fn_fused(int nImg, int nIfm, int nOfm,
	int ifhp, int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float pad_gemm_input[nImg][nIfm / GEMM_BLOCK][ifhp + 2 * pad_h][ifwp + 2 * pad_w][GEMM_BLOCK], float output[nImg][nOfm / GEMM_BLOCK][ofhp][ofwp][GEMM_BLOCK], const float filter[nOfm / GEMM_BLOCK][nIfm / GEMM_BLOCK][kh][kw][GEMM_BLOCK][GEMM_BLOCK], int iters)
{
	/* loop counters */
	int img, ofm_tile, ofm, ifm_tile, ifm, oj, oi, ij, ii, kj, ki, i;

#pragma scop
#pragma omp parallel for private(ofm_tile, ifm_tile, ij, oj, kj, ki, ii, oi, ifm, ofm)
	for (img = 0; img < nImg; ++img) {
		for (ofm_tile = 0; ofm_tile < nOfm / GEMM_BLOCK; ++ofm_tile) {
			for (ifm_tile = 0; ifm_tile < nIfm / GEMM_BLOCK; ++ifm_tile) {
				for (oj = 0; oj < ofh; ++oj) {
					ij = oj * STRIDE_H;

					for (kj = 0; kj < kh; ++kj) {
						for (ki = 0; ki < kw; ++ki) {
							//GEMM
							for (oi = 0; oi < ofw; ++oi) {
								ii = oi * STRIDE_W;
								for (ofm = 0; ofm < GEMM_BLOCK; ++ofm) {
									for (ifm = 0; ifm < GEMM_BLOCK; ++ifm) {
										output[img][ofm_tile][oj][oi][ofm] +=
											filter[ofm_tile][ifm_tile][kj][ki][ifm][ofm] *
											pad_gemm_input[img][ifm_tile][ij + kj][ii + ki][ifm];
									}
								}
							}
						}
					}
				}
			}

			for (oj = 0; oj < ofh; ++oj) {
				for (oi = 0; oi < ofw; ++oi) {
					for (ofm = 0; ofm < GEMM_BLOCK; ++ofm) {
						output[img][ofm_tile][oj][oi][ofm] =
							(output[img][ofm_tile][oj][oi][ofm] < 0.0f) ? 0.0f :
							output[img][ofm_tile][oj][oi][ofm];
					}
				}
			}

		}
	}
#pragma endscop
}

inline void padded_conv_relu_fp_libxsmm_core_gemm(int nImg, int nIfm, int nOfm, int ifhp,
	int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float pad_gemm_input[nImg][nIfm / GEMM_BLOCK][ifhp + 2 * pad_h][ifwp + 2 * pad_w][GEMM_BLOCK], float output[nImg][nOfm / GEMM_BLOCK][ofhp][ofwp][GEMM_BLOCK], const float filter[nOfm / GEMM_BLOCK][nIfm / GEMM_BLOCK][kh][kw][GEMM_BLOCK][GEMM_BLOCK], int iters)
{
	/* loop counters */
	int img, ofm_tile, ofm, ifm_tile, ifm, oj, oi, ij, ii, kj, ki, i;

#pragma omp parallel for private(ofm_tile, ifm_tile, ij, oj, kj, ki, ii)
	for (img = 0; img < nImg; ++img) {
		// zero_buf(&output[img][0][0][0][0], nOfm*ofhp*ofwp);
		// printf("thread id = %d\n", omp_get_thread_num());
		// #pragma omp parallel for private(ofm_tile, ifm_tile, oj, kj, ki)
		for (ofm_tile = 0; ofm_tile < nOfm / GEMM_BLOCK; ++ofm_tile) {
			for (ifm_tile = 0; ifm_tile < nIfm / GEMM_BLOCK; ++ifm_tile) {
				for (oj = 0; oj < ofh; ++oj) {
					ij = oj * stride_h;
					for (kj = 0; kj < kh; ++kj) {
						for (ki = 0; ki < kw; ++ki) {
							fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0],
								&pad_gemm_input[img][ifm_tile][ij + kj][ki][0],
								&output[img][ofm_tile][oj][0][0]);
						}
					}
				}
			}
		}
	}


#pragma omp parallel for private(ofm_tile, ij, oj, oi, ii, ofm)
	for (img = 0; img < nImg; ++img) {
		for (ofm_tile = 0; ofm_tile < nOfm / GEMM_BLOCK; ++ofm_tile) {
			for (oj = 0; oj < ofh; ++oj) {
				ij = oj * stride_h;
				for (oi = 0; oi < ofw; ++oi) {
					ii = oi * stride_w;
					for (ofm = 0; ofm < GEMM_BLOCK; ++ofm) {
						output[img][ofm_tile][oj][oi][ofm] *=
							( 1/ (1 + exp(-output[img][ofm_tile][oj][oi][ofm])) );
					}
				}
			}
		}
	}

}

inline void padded_conv_relu_fp_libxsmm_core_gemm_fused(int nImg, int nIfm, int nOfm, int ifhp,
	int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float pad_gemm_input[nImg][nIfm / GEMM_BLOCK][ifhp + 2 * pad_h][ifwp + 2 * pad_w][GEMM_BLOCK], float output[nImg][nOfm / GEMM_BLOCK][ofhp][ofwp][GEMM_BLOCK], const float filter[nOfm / GEMM_BLOCK][nIfm / GEMM_BLOCK][kh][kw][GEMM_BLOCK][GEMM_BLOCK], int iters)
{
	/* loop counters */
	int img, ofm_tile, ofm, ifm_tile, ifm, oj, oi, ij, ii, kj, ki, i;

#pragma omp parallel for private(ofm_tile, ifm_tile, ij, oj, kj, ki, ii, oi, ofm)
	for (img = 0; img < nImg; ++img) {
		for (ofm_tile = 0; ofm_tile < nOfm / GEMM_BLOCK; ++ofm_tile) {
			for (ifm_tile = 0; ifm_tile < nIfm / GEMM_BLOCK; ++ifm_tile) {
				for (oj = 0; oj < ofh; ++oj) {
					ij = oj * stride_h;
					for (kj = 0; kj < kh; ++kj) {
						for (ki = 0; ki < kw; ++ki) {
							fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0],
								&pad_gemm_input[img][ifm_tile][ij + kj][ki][0],
								&output[img][ofm_tile][oj][0][0]);
						}
					}
				}
			}

			for (oj = 0; oj < ofh; ++oj) {
				for (oi = 0; oi < ofw; ++oi) {
					for (ofm = 0; ofm < GEMM_BLOCK; ++ofm) {
						output[img][ofm_tile][oj][oi][ofm] *=
							( 1/ (1 + exp(-output[img][ofm_tile][oj][oi][ofm])) );
					}
				}
			}
		}
	}
}

inline void padded_conv_relu_fp_libxsmm_core_gemm_fused2(int nImg, int nIfm, int nOfm, int ifhp,
	int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float pad_gemm_input[nImg][nIfm / GEMM_BLOCK][ifhp + 2 * pad_h][ifwp + 2 * pad_w][GEMM_BLOCK], float output[nImg][nOfm / GEMM_BLOCK][ofhp][ofwp][GEMM_BLOCK], const float filter[nOfm / GEMM_BLOCK][nIfm / GEMM_BLOCK][kh][kw][GEMM_BLOCK][GEMM_BLOCK], int iters)
{
	/* loop counters */
	int img, ofm_tile, ofm, ifm_tile, ifm, oj, oi, ij, ii, kj, ki, i;

#pragma omp parallel for private(ofm_tile, ifm_tile, ij, oj, kj, ki, ii, oi, ofm)
	for (img = 0; img < nImg; ++img) {
		for (ofm_tile = 0; ofm_tile < nOfm / GEMM_BLOCK; ++ofm_tile) {

			for (ifm_tile = 0; ifm_tile < (nIfm / GEMM_BLOCK) - 1; ++ifm_tile) {
				for (oj = 0; oj < ofh; ++oj) {
					ij = oj * stride_h;
					for (kj = 0; kj < kh; ++kj) {
						for (ki = 0; ki < kw; ++ki) {
							fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0],
								&pad_gemm_input[img][ifm_tile][ij + kj][ki][0],
								&output[img][ofm_tile][oj][0][0]);
						}
					}
				}
			}

			for (ifm_tile = (nIfm / GEMM_BLOCK) - 1; ifm_tile < nIfm / GEMM_BLOCK; ++ifm_tile) {
				for (oj = 0; oj < ofh; ++oj) {
					ij = oj * stride_h;
					for (kj = 0; kj < kh; ++kj) {
						for (ki = 0; ki < kw; ++ki) {
							fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0],
								&pad_gemm_input[img][ifm_tile][ij + kj][ki][0],
								&output[img][ofm_tile][oj][0][0]);
						}
					}

					for (oi = 0; oi < ofw; ++oi) {
						for (ofm = 0; ofm < GEMM_BLOCK; ++ofm) {
							output[img][ofm_tile][oj][oi][ofm] *=
							( 1/ (1 + exp(-output[img][ofm_tile][oj][oi][ofm])) );
						}
					}
				}
			}
		}
	}
}


inline void padded_conv_relu_fp_libxsmm_core_gemm_fused3(int nImg, int nIfm, int nOfm, int ifhp,
	int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float pad_gemm_input[nImg][nIfm / GEMM_BLOCK][ifhp + 2 * pad_h][ifwp + 2 * pad_w][GEMM_BLOCK], float output[nImg][nOfm / GEMM_BLOCK][ofhp][ofwp][GEMM_BLOCK], const float filter[nOfm / GEMM_BLOCK][nIfm / GEMM_BLOCK][kh][kw][GEMM_BLOCK][GEMM_BLOCK], int iters)
{
	/* loop counters */
	int img, ofm_tile, ofm, ifm_tile, ifm, oj, oi, ij, ii, kj, ki, i;

#pragma omp parallel for private(ofm_tile, ifm_tile, ij, oj, kj, ki, ii, oi, ofm)
	for (img = 0; img < nImg; ++img) {
		for (ofm_tile = 0; ofm_tile < nOfm / GEMM_BLOCK; ++ofm_tile) {

			for (ifm_tile = 0; ifm_tile < (nIfm / GEMM_BLOCK) - 1; ++ifm_tile) {
				for (oj = 0; oj < ofh; ++oj) {
					ij = oj * stride_h;
					for (kj = 0; kj < kh; ++kj) {
						for (ki = 0; ki < kw; ++ki) {
							fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0],
								&pad_gemm_input[img][ifm_tile][ij + kj][ki][0],
								&output[img][ofm_tile][oj][0][0]);
						}
					}
				}
			}

			for (ifm_tile = (nIfm / GEMM_BLOCK) - 1; ifm_tile < nIfm / GEMM_BLOCK; ++ifm_tile) {
				for (oj = 0; oj < ofh; ++oj) {
					ij = oj * stride_h;

					for (kj = 0; kj < kh - 1; ++kj) {
						for (ki = 0; ki < kw; ++ki) {
							fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0],
								&pad_gemm_input[img][ifm_tile][ij + kj][ki][0],
								&output[img][ofm_tile][oj][0][0]);
						}
					}

					for (kj = kh - 1; kj < kh; ++kj) {
						for (ki = 0; ki < kw; ++ki) {
							fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0],
								&pad_gemm_input[img][ifm_tile][ij + kj][ki][0],
								&output[img][ofm_tile][oj][0][0]);
						}

						for (oi = 0; oi < ofw; ++oi) {
							for (ofm = 0; ofm < GEMM_BLOCK; ++ofm) {
								output[img][ofm_tile][oj][oi][ofm] =
									(output[img][ofm_tile][oj][oi][ofm] < 0.0f) ? 0.0f :
									output[img][ofm_tile][oj][oi][ofm];
							}
						}
					}

				}
			}
		}
	}
}

inline void padded_conv_relu_fp_libxsmm_core_gemm_fused4(int nImg, int nIfm, int nOfm, int ifhp,
	int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float pad_gemm_input[nImg][nIfm / GEMM_BLOCK][ifhp + 2 * pad_h][ifwp + 2 * pad_w][GEMM_BLOCK], float output[nImg][nOfm / GEMM_BLOCK][ofhp][ofwp][GEMM_BLOCK], const float filter[nOfm / GEMM_BLOCK][nIfm / GEMM_BLOCK][kh][kw][GEMM_BLOCK][GEMM_BLOCK], int iters)
{
	/* loop counters */
	int img, ofm_tile, ofm, ifm_tile, ifm, oj, oi, ij, ii, kj, ki, i;

#pragma omp parallel for private(ofm_tile, ifm_tile, ij, oj, kj, ki, ii, oi, ofm)
	for (img = 0; img < nImg; ++img) {
		for (ofm_tile = 0; ofm_tile < nOfm / GEMM_BLOCK; ++ofm_tile) {
			for (ifm_tile = 0; ifm_tile < nIfm / GEMM_BLOCK; ++ifm_tile) {
				for (oj = 0; oj < ofh; ++oj) {
					ij = oj * stride_h;
					for (kj = 0; kj < kh; ++kj) {
						for (ki = 0; ki < kw; ++ki) {
							fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0],
								&pad_gemm_input[img][ifm_tile][ij + kj][ki][0],
								&output[img][ofm_tile][oj][0][0]);

							if ((ifm_tile == (nIfm / GEMM_BLOCK) - 1) &&
								kj == kh - 1 && ki == kw - 1) {
								for (oi = 0; oi < ofw; ++oi) {
									for (ofm = 0; ofm < GEMM_BLOCK; ++ofm) {
										output[img][ofm_tile][oj][oi][ofm] =
											(output[img][ofm_tile][oj][oi][ofm] < 0.0f) ? 0.0f :
											output[img][ofm_tile][oj][oi][ofm];
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

inline void padded_conv_relu_fp_libxsmm_core_gemm_fused_swish(int nImg, int nIfm, int nOfm, int ifhp,
	int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float pad_gemm_input[nImg][nIfm / GEMM_BLOCK][ifhp + 2 * pad_h][ifwp + 2 * pad_w][GEMM_BLOCK], float output[nImg][nOfm / GEMM_BLOCK][ofhp][ofwp][GEMM_BLOCK], const float filter[nOfm / GEMM_BLOCK][nIfm / GEMM_BLOCK][kh][kw][GEMM_BLOCK][GEMM_BLOCK], int iters)
{
	/* loop counters */
		int img, ofm_tile, ofm, ifm_tile, ifm, oj, oi, ij, ii, kj, ki, i;


#pragma omp parallel for private(ofm_tile, ifm_tile, oj, kj, ki, ij, ii)
	for (img = 0; img < nImg; ++img) {
		// zero_buf(&output[img][0][0][0][0], nOfm*ofhp*ofwp);
		for (ofm_tile = 0; ofm_tile < nOfm / GEMM_BLOCK; ++ofm_tile) {
			for (kj = 0; kj < kh - 1; ++kj) {
				for (ki = 0; ki < kw; ++ki) {
					for (ifm_tile = 0; ifm_tile < nIfm / GEMM_BLOCK; ++ifm_tile) {
						for (oj = 0; oj < ofh; ++oj) {
							ij = oj * stride_h;

							fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0],
								&pad_gemm_input[img][ifm_tile][ij + kj][ki][0],
								&output[img][ofm_tile][oj][0][0]);

						}
					}
				}
			}
			for (kj = kh - 1; kj < kh; ++kj) {
				for (ki = 0; ki < kw - 1; ++ki) {
					for (ifm_tile = 0; ifm_tile < nIfm / GEMM_BLOCK; ++ifm_tile) {
						for (oj = 0; oj < ofh; ++oj) {
							ij = oj * stride_h;

							fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0],
								&pad_gemm_input[img][ifm_tile][ij + kj][ki][0],
								&output[img][ofm_tile][oj][0][0]);

						}
					}
				}
				for (ki = kw - 1; ki < kw; ++ki) {
					for (ifm_tile = 0; ifm_tile < (nIfm / GEMM_BLOCK)-1; ++ifm_tile) {
						for (oj = 0; oj < ofh; ++oj) {
							ij = oj * stride_h;

							fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0],
								&pad_gemm_input[img][ifm_tile][ij + kj][ki][0],
								&output[img][ofm_tile][oj][0][0]);

						}
					}
					for (ifm_tile = (nIfm / GEMM_BLOCK)-1; ifm_tile < nIfm / GEMM_BLOCK; ++ifm_tile) {
						for (oj = 0; oj < ofh; ++oj) {
							ij = oj * stride_h;

							fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0],
								&pad_gemm_input[img][ifm_tile][ij + kj][ki][0],
								&output[img][ofm_tile][oj][0][0]);
							
							for (oi = 0; oi < ofw; ++oi) {
								for (ofm = 0; ofm < GEMM_BLOCK; ++ofm) {
								output[img][ofm_tile][oj][oi][ofm] *=
								( 1/ (1 + exp(-output[img][ofm_tile][oj][oi][ofm])) );
								}
							}

						}
					}


				}


			}

		}
	}
}

inline void padded_conv_relu_fp_libxsmm_core_gemm_fused_swish2(int nImg, int nIfm, int nOfm, int ifhp,
	int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float pad_gemm_input[nImg][nIfm / GEMM_BLOCK][ifhp + 2 * pad_h][ifwp + 2 * pad_w][GEMM_BLOCK], float output[nImg][nOfm / GEMM_BLOCK][ofhp][ofwp][GEMM_BLOCK], const float filter[nOfm / GEMM_BLOCK][nIfm / GEMM_BLOCK][kh][kw][GEMM_BLOCK][GEMM_BLOCK], int iters)
{
	/* loop counters */
	int img, ofm_tile, ofm, ifm_tile, ifm, oj, oi, ij, ii, kj, ki, i;


#pragma omp parallel for private(ofm_tile, ifm_tile, oj, kj, ki, ij, ii)
	for (img = 0; img < nImg; ++img) {
		// zero_buf(&output[img][0][0][0][0], nOfm*ofhp*ofwp);
		for (oj = 0; oj < ofh; ++oj) {
			ij = oj * stride_h;
			for (kj = 0; kj < kh-1; ++kj) {
				for (ki = 0; ki < kw; ++ki) {
					for (ofm_tile = 0; ofm_tile < nOfm / GEMM_BLOCK; ++ofm_tile) {
						for (ifm_tile = 0; ifm_tile < nIfm / GEMM_BLOCK; ++ifm_tile) {

							fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0],
								&pad_gemm_input[img][ifm_tile][ij + kj][ki][0],
								&output[img][ofm_tile][oj][0][0]);
						}
					}
				}
			}
			for (kj = kh-1; kj < kh; ++kj) {
				for (ki = 0; ki < kw -1; ++ki) {
					for (ofm_tile = 0; ofm_tile < nOfm / GEMM_BLOCK; ++ofm_tile) {
						for (ifm_tile = 0; ifm_tile < nIfm / GEMM_BLOCK; ++ifm_tile) {

							fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0],
								&pad_gemm_input[img][ifm_tile][ij + kj][ki][0],
								&output[img][ofm_tile][oj][0][0]);
						}
					}
				}
				for (ki = kw -1; ki < kw; ++ki) {
					for (ofm_tile = 0; ofm_tile < nOfm / GEMM_BLOCK; ++ofm_tile) {
						for (ifm_tile = 0; ifm_tile < nIfm / GEMM_BLOCK; ++ifm_tile) {

							fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0],
								&pad_gemm_input[img][ifm_tile][ij + kj][ki][0],
								&output[img][ofm_tile][oj][0][0]);
						}

						for (oi = 0; oi < ofw; ++oi) {
							for (ofm = 0; ofm < GEMM_BLOCK; ++ofm) {
							output[img][ofm_tile][oj][oi][ofm] *=
							( 1/ (1 + exp(-output[img][ofm_tile][oj][oi][ofm])) );
							}
						}

					}
				}

			}
		}
	}	
}