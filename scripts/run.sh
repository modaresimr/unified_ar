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

for ((count=$1; count!=0; count--)); do
   echo "Running for the $count time======================="


for i in 1 2 3 4
do
	for seg in 0 1 2
	do
		echo start working on dataset $i and segmentation $seg
		OPENBLAS_MAIN_FREE=1 python main.py -d $i -s $seg -st 0&
#		PIDs="${PIDs} $!"
#		echo ${PIDS}
		sleep 2
	done

	echo finish for dataset $i
done
wait

done
