#!/bin/bash
nvidia-docker run \
  -v /home/jacob/github/training/speech_recognition:/home/jacob/github/training/speech_recognition:rw \
  -v /etc/passwd:/etc/passwd:ro \
  -it --rm --user $(id -u) ds2-cuda9cudnn7:gpu
  