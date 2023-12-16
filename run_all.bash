#!/bin/bash

if [ ! -d "terrain" ]; then
  echo "Bootstrap -- creating directories and generating terrain"
  mkdir "terrain"
  mkdir "results"
  python gen_terrain.py
fi

python main.py --terrain zeros
python main.py --terrain trapezoid
python main.py --terrain sawtooth
python main.py --terrain square
