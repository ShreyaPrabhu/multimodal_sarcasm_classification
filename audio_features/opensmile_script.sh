#!/bin/bash

audios_dir='../data/audios/utterances_final_wav'

for f in $audios_dir/*.wav; do
    echo $f
    if [[ ! -f "${f%.*}.csv" ]]; then
        ./opensmile-3.0.1-macos-x64/bin/SMILExtract -C opensmile-3.0.1-macos-x64/config/prosody/prosodyShs.conf -I $f -O "${f%.*}_prosody.csv"
    fi
done