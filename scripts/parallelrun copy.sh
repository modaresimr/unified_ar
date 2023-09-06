onINT() {
	read -n1 -p "Kill runnings? [y,n]" doit 
	case $doit in
	  y|Y) pkill -P $$ ;;
	  n|N) echo no ;;
	  *) echo dont know ;; 
	esac
        exit
}

trap "onINT" SIGINT
export logdir=logs/$(date +%Y-%m-%d/%H-%M-%S)
mkdir -p $logdir

GPUCOUNT=8
data='4'
comment='ali'
seg='3' #' 1 2 3'
feat='0'
# for s in `seq 1 1 6`;do
#   for o in `seq 1 2 $s`;do
#     seg_param="$seg_param size=$s,shift=$o"
#   done
# done
# for s in 5 `seq 10 10 30`;do
#   for o in 1 2 `seq 5 5 $s`;do
#   	seg_param="$seg_param size=$s,shift=$o"
#   done
# done
# for s in `seq 20 10 40`;do
#   for o in `seq 5 10 $s`;do
#   	seg_param="$seg_param size=$s,shift=$o"
#   done
# done
for s in `seq 10 10 120`;do
  for o in 5 `seq 10 10 $s`;do
  	seg_param="$seg_param size=$s,shift=$o"
  done
done
# seg_param=s=1
echo $seg_param
# for s in `seq 10 10 120`;do
#   	seg_param="$seg_param size=$s,shift=$s"
# 	echo '' >/dev/null
# done

# CUDA_VISIBLE_DEVICES=7 python main.py -c ali -d 0 -s 1 -f 0 -st 0&
# data='0 1 2'

parallel --eta  -j $GPUCOUNT 'echo run {1} on data {2} and segm {3} with param {4} feat={5}; TF_CPP_MIN_LOG_LEVEL=3 GOTOBLAS_MAIN_FREE=1 CUDA_VISIBLE_DEVICES=$[{%}-1] python main.py -c {1} -d {2} -s {3} -sp {4} -f {5}  -st 0 > $logdir/c={1}-d={2}-s={3}-sp-{4}-f={5}-st=0 2> >(tee -a $logdir/c={1}-d={2}-s={3}-f={5}-st=0 >&2)' ::: $comment :::  $data  ::: $seg ::: $seg_param ::: $feat

