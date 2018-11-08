#!/bin/bash
#${1} = python version... use .py3 for python 3
nvidia-docker build . --rm -f Dockerfile.gpu${1} -t ds2-cuda9cudnn${1}:gpu
