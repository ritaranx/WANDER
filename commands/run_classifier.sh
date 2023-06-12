task=mesh 
gpu=1
n_gpu=2

train_seed=42
model_type=scibert 
semi_method=ft
method=train
max_seq_len=256
max_seq_len_test=256
eval_batch_size=64
dev_labels=100
steps=100
logging_steps=25
st_logging_steps=25
k_cal=5
dataset=${task}
dr_model=arxiv_ckpt 
topN=100
round=6
loc=1
glob=1
train_file="${dataset}_${dr_model}_train_top${topN}_round${round}.jsonl"
suffix="${dr_model}_N${topN}_loc${loc}_global${glob}"

lr=2e-5
batch_size=32
epochs=8
load_prev_cmd="--load_prev=0 --prev_ckpt=${CKPT_OF_PREVIOUS_ROUNDS}" # to be filled


eps=0.8
self_training_weight=1
gce_loss_q=0.6
lr_st=2e-6
self_training_batch_size=32
self_training_update_period=200
self_training_max_step=1000
num_unlabeled=9000
ssl_cmd="--learning_rate_st=${lr_st} --self_training_eps=${eps} --self_training_weight=${self_training_weight} --self_training_update_period=${self_training_update_period} --gce_loss_q=${gce_loss_q} --num_unlabeled=${num_unlabeled}"

output_dir=datasets/${task}/model 
mkdir -p ${output_dir}
echo ${method}
mkdir -p ${task}/cache/loc${loc}_glob${glob}

train_cmd="CUDA_VISIBLE_DEVICES=${gpu} python3 main.py --do_train --do_eval --task=${task} \
	--train_file=${train_file} --dev_file=dev.jsonl --test_file=test.jsonl --unlabel_file=train.jsonl \
	--dr_model=${dr_model} --dr_N=${topN} \
	--data_dir=../datasets/${task}_openws/${suffix} --eval_dir=../datasets/${task} --train_seed=${train_seed} \
	--cache_dir="${task}/cache/loc${loc}_glob${glob}/" --output_dir=${output_dir} \
	--logging_steps=${logging_steps} --self_train_logging_steps=${st_logging_steps} --dev_labels=${dev_labels} \
	--gpu=${gpu} --n_gpu=${n_gpu} --num_train_epochs=${epochs} --weight_decay=1e-8 --learning_rate=${lr}  \
	--method=${method} --batch_size=${batch_size} --eval_batch_size=${eval_batch_size} \
	--self_training_batch_size=${self_training_batch_size} \
	--max_seq_len=${max_seq_len} --max_seq_len_test=${max_seq_len_test} --auto_load=1 \
	--max_steps=${steps} --model_type=${model_type} \
	--self_training_max_step=${self_training_max_step} ${load_prev_cmd} \
	--semi_method=${semi_method} ${ssl_cmd} --k_cal=${k_cal} --round=${round}"
echo $train_cmd
eval $train_cmd
