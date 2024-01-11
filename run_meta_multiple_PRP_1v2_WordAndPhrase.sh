task=$1 #citation_intent
outdir=$2 #exist_method
device=$3 #'0,1,2,3'
start=$4 #0
end=$5 #3
metric=$6 #f1 # Use {accuracy} for chemprot and hyperpartisan and {f1} for citation_intent and sciie
mlm_probability=$7
pip3 install allennlp overrides pytorch_pretrained_bert matplotlib
python3 -m spacy download en_core_web_sm
# pip3 install requests==2.27.1    
# pip3 show en_core_web_sm
# pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.3.0/en_core_web_sm-3.3.0.tar.gz
# pip3 install spacy==3.3.3

# Generate elements for matrices A and B 0.2に固定
A_elements=(1.0) # v1(maskonly): 0.2 , v2(mask and random): 1, v3: 0.4, v4: 1.0で分類が過学習しているのでdropout 0.2 に倍増, v5: PLLスコアを正規化した値をマスク確率にした, v6: masking_probabilityを平均の閾値をなくした, -1~1の正規化ではなく，0~1にしてk倍, v6.1:k倍を消した．フレーズスコアの算出の際，単語内のサブワードスコアの合計を取るのを忘れている, v6.1:定義通りに修正
# $(seq 0 0.2 0.2)
B_elements=(1.0)
# $(seq 0 0.2 0.2)

# Initialize variables to store the combinations
combined_A=""
combined_B=""
unique_strings=""

# Counter for generating unique strings
counter=0

# Generate all combinations
for a in $A_elements; do
    for b in $B_elements; do
        combined_A+="$a "
        combined_B+="$b "
        unique_strings+="TAPT_MLMPRP_$counter "
        ((counter++))
    done
done

# Remove the trailing space
combined_A=${combined_A% }
combined_B=${combined_B% }



unique_strings=${unique_strings% }

# Output the results (optional)
echo "A elements: $combined_A"
echo "B elements: $combined_B"
echo "Unique strings: $unique_strings"

# Storing the results in variables
matrix_A="$combined_A"
matrix_B="$combined_B"
unique_strs="$unique_strings"

# Calculate the number of elements in matrix A
num_elementsA=$(echo "$A_elements" | wc -w)
num_elementsB=$(echo "$B_elements" | wc -w)
num_elements=$num_elementsA*$num_elementsB

echo "要素数 $num_elements"

# Initialize the variable for storing paths
paths=""

# Generate the paths equal to the number of elements in matrix A
for (( i=0; i<$num_elements; i++ )); do
    paths+="datasets/$task/train.txt "
done

# Remove the trailing space
paths=${paths% }

# Output the paths (optional)
echo "Paths: $paths"

# Storing the paths in a variable
train_paths="$paths"

# IFS=',' read -r -a device_array <<< "$device_string"

# length=${#IFS[@]}
# nvidia-smi

devices=($device)
length=${#devices[@]}


seeds=($(seq $start 1 $end))
echo 'seeds: '${seeds}


# for k in $(seq $start $end)
for ((i = 0; i < length; i++))
do
	echo 'GPU_number='${devices[$i]}

    echo 'Working on Seed='${seeds[$i]}
    mkdir -p $outdir'_True1_v6_4'/$task/meta_tartan/seed=${seeds[$i]}
    python3 -u -m scripts.run_mlm_auxiliary_Change_PLLMask --mlm_mask 0.8 --mlm_random 0.5 --mlm_probability 0.15 --train_data_file $paths --aux-task-names $unique_strs --word_mask_probability $matrix_A --phrase_mask_probability $matrix_B --line_by_line --model_type roberta-base --tokenizer_name roberta-base --mlm --model_name_or_path roberta-base --do_train --learning_rate 1e-4 --block_size 128 --logging_steps 5000 --classf_lr 1e-3 --primary_task_id $task -weight-strgy 'meta' --classifier_dropout 0.1 --classf_ft_lr 5e-6 --classf_max_seq_len 128 --classf-metric $metric --lazy-dataset --output_dir $outdir'_True1_v6_4'/$task/meta_tartan/seed=${seeds[$i]} --overwrite_output_dir --seed ${seeds[$i]}  --classf_dev_wd 0.1 --classf_dev_lr 1e-3 --meta-lr-weight 1e-2 --dev_batch_sz 32 --classf_patience 20 --num_train_epochs 150 --classf_iter_batchsz 64 --per_gpu_train_batch_size 64 --gradient_accumulation_steps 2 --eval_every 30 --tapt-primsize --classf_warmup_frac 0.06 &> $outdir'_True1_v6_4'/$task/meta_tartan/seed=${seeds[$i]}.txt
done
#CUDA_VISIBLE_DEVICES=$device 

#batchsz 64→64
#v4 classifier_dropout 0.1→0.2
#v5 classifier_dropout 0.1
#v6_1 kを消した