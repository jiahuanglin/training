# Script to train and time DeepSpeech 2 implementation

RANDOM_SEED=1
TARGET_ACC=23

#${1} = the epoch number of the model you want to convert and then test

python3 convert_torch_to_onnx.py --checkpoint --continue_from models/deepspeech_${1}.pth.tar --model_path models/deepspeech_t$RANDOM_SEED.pth.tar --seed $RANDOM_SEED --acc $TARGET_ACC 

python3 test_onnx.py ${1}
