for i in {0..0}; do
    python pretrain.py \
        --epochs 200 \
        --batch_size 96 \
        --weight_decay 0.1 \
        --ntoken 15 \
        --nclass 5 \
        --ninp 256 \
        --nhead 32 \
        --nhid 1024 \
        --nlayers 5 \
        --nmute 0 \
        --dropout 0.1 \
        --kmers 5 \
        --kmer_aggregation \
        --lr_scale 0.1 \
        --warmup_steps 600 \
        --fold $i \
        --gpu_id 0 \
        --path data \
        --workers 2 
done
