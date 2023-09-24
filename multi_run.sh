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
data='2'
comment='comment'
seg='3' #' 1 2 3'
feat='0'
for s in 1 2 5 15`seq 10 10 60`;do
  for o in 1 2 5 15 `seq 10 10 $s`;do
  	seg_param="$seg_param size=$s,shift=$o"
  done
done
# seg_param=s=1
echo $seg_param

parallel --eta  -j $GPUCOUNT 'echo run {1} on data {2} and segm {3} with param {4} feat={5}; TF_CPP_MIN_LOG_LEVEL=3 GOTOBLAS_MAIN_FREE=1 CUDA_VISIBLE_DEVICES=$[{%}-1] python main.py -c {1} -d {2} -s {3} -sp {4} -f {5}  -st 0 > $logdir/c={1}-d={2}-s={3}-sp-{4}-f={5}-st=0 2> >(tee -a $logdir/c={1}-d={2}-s={3}-f={5}-st=0 >&2)' ::: $comment :::  $data  ::: $seg ::: $seg_param ::: $feat

