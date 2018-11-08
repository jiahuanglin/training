# Script to train and time DeepSpeech 2 implementation

RANDOM_SEED=1
TARGET_ACC=23

# ${1} = the model to use
# ${2} = the test set to use (libri, ov)
# ${3} = the hold idx
# ${4} = the duration of the held input
# ${5} = number of trial, set to -1 for no cap on number of trials

# run the same hold idx for cpu, gpu and varying batch sizes
python test.py --cpu 0 --batch_size_val 40 --checkpoint --continue_from models/deepspeech_${1}.pth.tar --use_set ${2} --seed $RANDOM_SEED --acc $TARGET_ACC --hold_idx ${3} --hold_sec ${4} --n_trials ${5}
exit 1
python test.py --cpu 0 --batch_size_val 64 --checkpoint --continue_from models/deepspeech_${1}.pth.tar --use_set ${2} --seed $RANDOM_SEED --acc $TARGET_ACC --hold_idx ${3} --hold_sec ${4} --n_trials ${5}
python test.py --cpu 0 --batch_size_val 4 --checkpoint --continue_from models/deepspeech_${1}.pth.tar --use_set ${2} --seed $RANDOM_SEED --acc $TARGET_ACC --hold_idx ${3} --hold_sec ${4} --n_trials ${5}
python test.py --cpu 0 --batch_size_val 48 --checkpoint --continue_from models/deepspeech_${1}.pth.tar --use_set ${2} --seed $RANDOM_SEED --acc $TARGET_ACC --hold_idx ${3} --hold_sec ${4} --n_trials ${5}
python test.py --cpu 0 --batch_size_val 12 --checkpoint --continue_from models/deepspeech_${1}.pth.tar --use_set ${2} --seed $RANDOM_SEED --acc $TARGET_ACC --hold_idx ${3} --hold_sec ${4} --n_trials ${5}
python test.py --cpu 0 --batch_size_val 32 --checkpoint --continue_from models/deepspeech_${1}.pth.tar --use_set ${2} --seed $RANDOM_SEED --acc $TARGET_ACC --hold_idx ${3} --hold_sec ${4} --n_trials ${5}
python test.py --cpu 0 --batch_size_val 24 --checkpoint --continue_from models/deepspeech_${1}.pth.tar --use_set ${2} --seed $RANDOM_SEED --acc $TARGET_ACC --hold_idx ${3} --hold_sec ${4} --n_trials ${5}
python test.py --cpu 1 --batch_size_val 1 --checkpoint --continue_from models/deepspeech_${1}.pth.tar --use_set ${2} --seed $RANDOM_SEED --acc $TARGET_ACC --hold_idx ${3} --hold_sec ${4} --n_trials ${5}
python test.py --cpu 1 --batch_size_val 64 --checkpoint --continue_from models/deepspeech_${1}.pth.tar --use_set ${2} --seed $RANDOM_SEED --acc $TARGET_ACC --hold_idx ${3} --hold_sec ${4} --n_trials ${5}
python test.py --cpu 1 --batch_size_val 4 --checkpoint --continue_from models/deepspeech_${1}.pth.tar --use_set ${2} --seed $RANDOM_SEED --acc $TARGET_ACC --hold_idx ${3} --hold_sec ${4} --n_trials ${5}
python test.py --cpu 1 --batch_size_val 48 --checkpoint --continue_from models/deepspeech_${1}.pth.tar --use_set ${2} --seed $RANDOM_SEED --acc $TARGET_ACC --hold_idx ${3} --hold_sec ${4} --n_trials ${5}
python test.py --cpu 1 --batch_size_val 12 --checkpoint --continue_from models/deepspeech_${1}.pth.tar --use_set ${2} --seed $RANDOM_SEED --acc $TARGET_ACC --hold_idx ${3} --hold_sec ${4} --n_trials ${5}
python test.py --cpu 1 --batch_size_val 32 --checkpoint --continue_from models/deepspeech_${1}.pth.tar --use_set ${2} --seed $RANDOM_SEED --acc $TARGET_ACC --hold_idx ${3} --hold_sec ${4} --n_trials ${5}
python test.py --cpu 1 --batch_size_val 24 --checkpoint --continue_from models/deepspeech_${1}.pth.tar --use_set ${2} --seed $RANDOM_SEED --acc $TARGET_ACC --hold_idx ${3} --hold_sec ${4} --n_trials ${5}
