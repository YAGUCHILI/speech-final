#!/bin/bash

CONFIG_FILE="config.sh"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
else
    echo "Error: config file not found: $CONFIG_FILE"
    exit 1
fi

if [ -z "$INPUT_WAV" ] || [ -z "$VOCAL_DIR" ] || [ -z "$ACCOM_DIR" ]; then
    echo "Error: INPUT_WAV, VOCAL_DIR, and ACCOM_DIR must be set in config.sh"
    exit 1
fi

mkdir -p "$VOCAL_DIR" "$ACCOM_DIR"

export weight_uvr5_root="${weight_uvr5_root:-assets/uvr5_weights}"
export UVR5_ACCOM_DIR="$ACCOM_DIR"

python run_uvr5_pipeline.py "$INPUT_WAV" "$VOCAL_DIR"
