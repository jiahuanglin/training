#!/bin/bash

nvidia-docker run \
  --shm-size 64G \
  -v /home/$USER/github/training/speech_recognition:/home/jacob/github/training/speech_recognition:rw \
  -v /scratch:/scratch:rw \
  -v /etc/passwd:/etc/passwd:ro \
  -p 127.0.0.1:5050:5050/tcp \
  -it --rm -u 0 ds2-cuda9cudnn7:gpu


  # -it --rm -u 0 ds2-cuda9cudnn7:gpu
