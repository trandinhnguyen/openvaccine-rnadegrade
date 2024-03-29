for i in {0..4}; do
    python train_pl.py --gpu_id 0 --kmer_aggregation --nmute 0 --epochs 75 --nlayers 5 \
        --batch_size 16 --kmers 5 --lr_scale 0.1 --path ../../input --workers 16 \
        --dropout 0.1 --nclass 5 --ntoken 15 --nhead 32 --ninp 256 --nhid 1024 --warmup_steps 600 \
        --fold $i --weight_decay 0.1 --nfolds 10 --error_alpha 0.5 --noise_filter 0.25
done
