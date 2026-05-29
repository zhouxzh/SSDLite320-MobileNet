#!/bin/bash

rsync -avz --exclude "checkpoints" --exclude "data" --exclude "viz_results"  4090D:/data/zhong/MobileNet-SSD/ ./

