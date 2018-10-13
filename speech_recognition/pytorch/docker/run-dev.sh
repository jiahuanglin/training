#!/bin/bash
nvidia-docker run \
  -v /home/jacob/github/training/speech_recognition:/home/jacob/github/training/speech_recognition:rw \
  -v /scratch:/scratch:rw \
  -v /etc/passwd:/etc/passwd:ro \
  -it --rm -u 0 ds2-cuda9cudnn7:gpu
