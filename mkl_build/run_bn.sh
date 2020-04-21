OUT=bn_mkl_perf.csv
set -x
#rm ${OUT}
BENCHDNN=/nfs_home/stavarag/work/polyfusion/mkl_build/mkl-dnn/build/tests/benchdnn/benchdnn

TAGS="--tag=nChw8c" #FIXME 

#Default values.
iters=1
ifw=56
ifh=56
nIfm=64
nOfm=256
kw=1
kh=1
pad_w=0
pad_h=0
stride=1

mb=28
#Taking values from terminal

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

oh=$(((ifh + 2*pad_h - kh)/stride + 1))
ow=$(((ifw + 2*pad_w - kw)/stride + 1))

for mb in 28
do

export KMP_AFFINITY=granularity=fine,compact,1,28
export OMP_NUM_THREADS=$mb


config="mb${mb}ic${nIfm}ih${ifh}iw${ifw}n"

#echo -n $config, >> ${OUT}
runtime=`$BENCHDNN --bnorm --mode=p --dir=FWD_I --dt=f32 --flags=GSR $TAGS  $config"resnet_50:conv1" | grep "nresnet_50:conv1"| cut -d"," -f6`
echo runtime: $runtime
FLOPS=$(((mb * nIfm * ifh * ifw * 4)))
echo -n ${config_num},$FLOPS,$runtime >> ${mb}_${OUT}
echo "" >> ${mb}_${OUT}


done
