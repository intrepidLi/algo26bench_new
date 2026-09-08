mkdir -p logs

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src nohup \
  torchrun --standalone --nproc_per_node=4 \
  -m algo26bench.bench.run --config configs/hyformer_150w_train_eval.json \
  > logs/hyformer_1epoch_$(date +%Y%m%d_%H%M%S).log 2>&1 &