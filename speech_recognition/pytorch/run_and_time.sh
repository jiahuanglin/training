# Script to train and time DeepSpeech 2 implementation

RANDOM_SEED=1
TARGET_ACC=23

python dist_train.py --world_size=2 --checkpoint --model_path models/deepspeech_t$RANDOM_SEED.pth.tar --seed $RANDOM_SEED --acc $TARGET_ACC
#python train.py --checkpoint --model_path models/deepspeech_t$RANDOM_SEED.pth.tar --seed $RANDOM_SEED --acc $TARGET_ACC
#python train.py --checkpoint --continue_from models/deepspeech_${1}.pth.tar --model_path models/deepspeech_t$RANDOM_SEED.pth.tar --seed $RANDOM_SEED --acc $TARGET_ACC
