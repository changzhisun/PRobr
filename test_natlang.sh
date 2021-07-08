data_dir=./data/NatLang
lr=1e-5
ep=50
eval_split=test
output_dir=./output/NatLang
# model=roberta_rr # PRover
model=roberta_probr  # PRobr
echo $output_dir

python3 ./run_experiment_natlang.py                        \
    --data_dir $data_dir                                             \
    --output_dir $output_dir                                         \
    --per_gpu_eval_batch_size  32                                     \
    --per_gpu_train_batch_size 8                                    \
    --model_type $model                                   \
    --model_name_or_path roberta-large                               \
    --task_name rr                                                   \
    --run_on_test                                                        \
    --do_lower_case                                                  \
    --max_seq_length 300                                             \
    --learning_rate $lr                                              \
    --gradient_accumulation_steps 1                                  \
    --num_train_epochs $ep                                             \
    --logging_steps 4752                                             \
    --save_steps 4750                                                \
    --seed 42                                                        \
    --data_cache_dir ./output/cache/                                 \
    --evaluate_during_training                                       \

python3 ./ilp_infer/inference_natlang.py                                      \
    --data_dir $data_dir                                             \
    --eval_split $eval_split                                         \
    --node_preds $output_dir/prediction_nodes_${eval_split}.lst                \
    --edge_logits $output_dir/prediction_edge_logits_${eval_split}.lst         \
    --edge_preds $output_dir/edge_preds_natlang_${eval_split}.lst                   \
    --natlang_metadata $data_dir/turk-questions-${eval_split}-mappings.tsv

python3 ./evaluation/eval_natlang.py                                    \
    --data_dir $data_dir                                             \
    --eval_split $eval_split                                         \
    --qa_pred_file $output_dir/predictions_${eval_split}.lst         \
    --node_pred_file $output_dir/prediction_nodes_${eval_split}.lst  \
    --edge_pred_file $output_dir/edge_preds_natlang_${eval_split}.lst     \
    --natlang_metadata $data_dir/turk-questions-${eval_split}-mappings.tsv  \
   | tee $output_dir/${eval_split}_natlang_eval.log
