# 定义GLUE_DIR: 代表微调模型数据所在的路径
export GLUE_DIR=./glue_data

# 定义OUT_DIR: 代表模型的保存路径
export OUT_DIR=./bert_finetuning_test/

# 编写shell执行脚本
python ./run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name MRPC \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/MRPC \
    --max_seq_length 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 1.0 \
    --output_dir $OUT_DIR

