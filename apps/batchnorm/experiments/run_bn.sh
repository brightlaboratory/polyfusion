

set -x
export KMP_AFFINITY=granularity=fine,compact,1,28
export LD_LIBRARY_PATH=/nfs_home/stavarag/work/software/barvinok/barvinok-0.41.2_install/lib:/nfs_home/stavarag/work/software/barvinok/isl_install/lib:$LD_LIBRARY_PATH

OUT=poly_perf.csv

check_correctness=1
PERF_DIR=perf_data
CONFIG_DIR=configs
TEMP=temp
VERSIONS='0 1'
DATATYPESIZE=4

mkdir ${PERF_DIR}
mkdir ${TEMP}
       iters=$1
       ifw=$2
       ifh=$3
       nIfm=$4
       nOfm=$5
       kw=$6
       kh=$7
       pad_w=$8
       pad_h=$9
       stride=${10}
       config_num=${11}


	for images in 28
	do
	        CONFIG_OUT=${PERF_DIR}/${config_num}_${images}_${OUT}
		META_CONFIG_OUT=${PERF_DIR}/meta_${config_num}_${images}_${OUT}
	        rm ${CONFIG_OUT}
		rm ${META_CONFIG_OUT}

		export OMP_NUM_THREADS=${images}
		for version in $VERSIONS #FIXME
		do
		  (cd .. && make clean && make MACROFLAGS="-DSH=$stride -DSTRIDE_W=$stride ") 	
		   ../bn_relu ${iters} ${images} ${ifw} ${ifh} ${nIfm} ${version} ${check_correctness} &> run_output
				GFLOPS=`cat run_output |  grep Real_GFLOPS |  cut -d= -f2`
				NAIVE_GFLOPS=`cat run_output |  grep Naive_GFLOPS |  cut -d= -f2`
				ERROR=`cat run_output | grep "inf-norm of comp. abs. error" | cut -d: -f 2`

                                { echo "${version},${GFLOPS}" ; } >> ${CONFIG_OUT}
				echo  "${NAIVE_GFLOPS},${ERROR}" >> ${META_CONFIG_OUT}
		done
	done

