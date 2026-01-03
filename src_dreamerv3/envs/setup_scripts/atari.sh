#!/bin/sh

# Run this script to install Atari (ALE + ROMs)
pip3 install "gymnasium[atari]"
pip3 install AutoROM
pip3 install opencv-python==4.7.0.72
AutoROM --accept-license
