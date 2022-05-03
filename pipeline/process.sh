#!/usr/bin/env bash

export dataset=$1
#python /home/franz/derain/methods/HINet/basicsr/run_folder.py --output_dir /home/franz/derain/out/${dataset}/qhinet/  --input_dir /home/franz/derain/data/${dataset}/input/ -opt /home/franz/derain/methods/QHINet/options/demo/demo.yml

#python3 RefineDNet-for-dehazing/quick_test.py  --input_dir /home/franz/derain/out/${dataset}/derained_2 --output_dir /home/franz/derain/out/${dataset}/dehazed_2 --dataset_mode single --name refined_DCP_outdoor --model refined_DCP --phase test --preprocess none --save_image --method_name refined_DCP_outdoor_ep_60 --epoch 60 --checkpoints_dir ./RefineDNet-for-dehazing/checkpoints
echo ${dataset}
python retinexnet/predict.py --gpu_id 0 --data_dir /home/franz/derain/out/${dataset}/qhinet720 --res_dir /home/franz/derain/out/${dataset}/lle --ckpt_dir /home/franz/derain/pipeline/retinexnet/ckpts
python sum.py --input_dir_lle /home/franz/derain/out/${dataset}/lle --input_dir_dehazed /home/franz/derain/out/${dataset}/qhinet720  --output_dir /home/franz/derain/out/${dataset}/lle_sume
python contrast.py --input_dir /home/franz/derain/out/${dataset}/lle_sume --output_dir /home/franz/derain/out/${dataset}/stretched --method stretch
python sharpen.py --input_dir /home/franz/derain/out/${dataset}/stretched --output_dir /home/franz/derain/out/${dataset}/sharpened
python grayworld.py --input_dir /home/franz/derain/out/${dataset}/sharpened --output_dir /home/franz/derain/out/${dataset}/color_corrected

