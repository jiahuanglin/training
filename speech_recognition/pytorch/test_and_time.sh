# Script to train and time DeepSpeech 2 implementation

RANDOM_SEED=1
TARGET_ACC=23

python test.py --cpu ${4} --checkpoint --continue_from models/deepspeech_${1}.pth.tar --use_set ${2} --seed $RANDOM_SEED --acc $TARGET_ACC --hold_idx ${3}
