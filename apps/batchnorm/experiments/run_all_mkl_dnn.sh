FILE=../../../mkl_build/run_bn.sh
set -x
sh run_resnet.sh $FILE
sh run_fastrcnn.sh $FILE
sh run_maskrcnn.sh $FILE
sh run_xception.sh $FILE
sh run_yolov2.sh $FILE
sh run_mobilenet.sh $FILE
sh run_alexnet.sh $FILE
sh run_overfeat.sh $FILE
sh run_googlenetv1.sh $FILE
sh run_googlenetv3.sh $FILE
sh run_vgga.sh $FILE
