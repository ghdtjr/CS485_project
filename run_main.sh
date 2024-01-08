#!/bin/bash

# Loop through the drop_out values
# for drop_out in $(seq 0 0.1 0.99)
# do
# 	for num_epochs in $(seq 10 10 200)
# 	do
#     	echo "Running main.py with drop_out=$drop_out"
#     	nohup python3 main.py --gpu_num 1 --dropout $drop_out --num_epochs $num_epochs
# 	done
# done

# ("--lr", type=float, default=0.0001)
# for lr in $(seq 0.00001 0.1 0.99)
# for lr in 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1;
# do
# 	# echo "Running main.py with drop_out=$drop_out"
# 	nohup python3 main.py --gpu_num 1 --lr $lr
# done

# for weight_decay in $(seq 0 0.1 0.99)
# do
# 	# echo "Running main.py with drop_out=$drop_out"
# 	nohup python3 main.py --gpu_num 1 --weight_decay $weight_decay
# done

# for batch_size in 2 4 8 16 32 64 128 256;
# do
# 	# echo "Running main.py with drop_out=$drop_out"
# 	nohup python3 main.py --gpu_num 1 --batch_size $batch_size
# done


# for loss_fun in "hinge";
# do
# 	# echo "Running main.py with drop_out=$drop_out"
# 	nohup python3 main.py --gpu_num 1 --loss_fun $loss_fun
# done

for target in "Compression";
do
	python3 main.py --gpu_num 1 --target $target
done


# for target in "Drop_out" 'Num_Epoch' 'Learning_Rate' 'Regularization' 'Batch_Size' 'Loss_function';
# do
# 	nohup python3 main.py --gpu_num 1 --target $target
# done