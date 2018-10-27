#!/bin/bash
#${1} = python version.. use .py3 for python3
nvidia-docker run \
  --shm-size 64G \
  --network host \
  -v /home/$USER/github/training/speech_recognition:/home/jacob/github/training/speech_recognition:rw \
  -v /scratch:/scratch:rw \
  -v /etc/passwd:/etc/passwd:ro \
  -p 5050:5050/tcp \
  -it --rm -u 0 ds2-cuda9cudnn7${1}:gpu


  # -it --rm -u 0 ds2-cuda9cudnn7:gpu
