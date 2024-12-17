#!/bin/bash

# Set MASTER_IP and NODE_NUM based on hostname
if [[ "$HOSTNAME" == *"master-0"* ]]; then
    # Remove existing master_ip file
    if [ -f $SHARED_PATH/master_ip ]; then
        rm $SHARED_PATH/master_ip
    fi
    
    MASTER_IP=$(ip -4 addr show ${NETWORK_INTERFACE} | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
    echo "MASTER_IP set to $MASTER_IP (Master Node)"
    echo $MASTER_IP > $SHARED_PATH/master_ip
    export NODE_NUM=0
else
    # Wait until the master IP is available in the shared path
    while [ ! -f $SHARED_PATH/master_ip ]; do
        echo "Waiting for master_ip file..."
        sleep 1
    done
    MASTER_IP=$(cat $SHARED_PATH/master_ip)
    export NODE_NUM=$(($(hostname | sed 's/[^0-9]*//g') + 1))
    echo "MASTER_IP set to $MASTER_IP (Worker Node)"
fi

# Construct and run the torchrun command
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --nnodes=${NNODES} \
         --node_rank=${NODE_NUM} \
         --master_addr=${MASTER_IP} \
         --master_port=${MASTER_PORT} \
         main.py \
         --backend=${BACKEND} \
         --batch_size=${BATCH_SIZE} \
         --data_path=${DATA_PATH} \
         --num_train_epochs=${NUM_TRAIN_EPOCHS} \
         --learning_rate=${LEARNING_RATE} \
         --num_workers=${NUM_WORKERS} \
         --print_interval=${PRINT_INTERVAL} \
	 --output_dir=${OUTPUT_DIR}
