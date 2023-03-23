#!/bin/bash
# ------------------------------------------------------------------
# [Author] Title
#          Description
# ------------------------------------------------------------------

VERSION=0.1.0
SUBJECT=DSTC11
USAGE="Usage: "

# Please clone https://github.com/alexa/alexa-with-dstc9-track1-dataset to download the data.
# git clone https://github.com/alexa/alexa-with-dstc9-track1-dataset
DSTC11_PATH=~/data/volume_2/dstc/dstc11-track5/data
python converter.py ${DSTC11_PATH}