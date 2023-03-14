DATASET="physics"
HOPS=5
LEARNING_RATE=0.01
WEIGHT_DECAY=0.00005
BATCH_SIZE=2000

NAME="DATASET:$DATASET+NHOP:$NHOP+LR:$LEARNING_RATE+WD:$WEIGHT_DECAY+BS:$BATCH_SIZE"

python -u train.py --name $NAME --dataset $DATASET --hops $HOPS --n_heads 8 --n_layers 1 \
                    --peak_lr $LEARNING_RATE --weight_decay $WEIGHT_DECAY --readout att-sum \
                    --batch_size $BATCH_SIZE --pe_dim=10 --isPE=1 --isHE=0 --isSUM=0 --hidden_dim=128





