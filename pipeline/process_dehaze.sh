export dataset=$1
#python /home/franz/derain/methods/HINet/basicsr/run_folder.py --output_dir /home/franz/derain/out/${dataset}/qhinet/  --input_dir /home/franz/derain/data/${dataset}/input/ -opt /home/franz/derain/methods/QHINet/options/demo/demo.yml

python RefineDNet-for-dehazing/quick_test.py  --input_dir /home/franz/derain/out/${dataset}/qhinet720 --output_dir /home/franz/derain/out/${dataset}/deqhinet720 --dataset_mode single --name refined_DCP_outdoor --model refined_DCP --phase test --preprocess none --save_image --method_name refined_DCP_outdoor_ep_60 --epoch 60 --checkpoints_dir ./RefineDNet-for-dehazing/checkpoints
python retinexnet/predict.py --gpu_id 0 --data_dir /home/franz/derain/out/${dataset}/deqhinet720 --res_dir /home/franz/derain/out/${dataset}/delle --ckpt_dir /home/franz/derain/pipeline/retinexnet/ckpts
python sum.py --input_dir_lle /home/franz/derain/out/${dataset}/delle --input_dir_dehazed /home/franz/derain/out/${dataset}/deqhinet720  --output_dir /home/franz/derain/out/${dataset}/delle_sume
python contrast.py --input_dir /home/franz/derain/out/${dataset}/delle_sume --output_dir /home/franz/derain/out/${dataset}/destretched --method stretch
python sharpen.py --input_dir /home/franz/derain/out/${dataset}/destretched --output_dir /home/franz/derain/out/${dataset}/desharpened
python grayworld.py --input_dir /home/franz/derain/out/${dataset}/desharpened --output_dir /home/franz/derain/out/${dataset}/decolor_corrected