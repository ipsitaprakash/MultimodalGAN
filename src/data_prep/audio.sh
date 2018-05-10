#!/bin/sh
c=20276 
while [ "$c" -gt 10 ] 
do
    tail -n "$c" audio_files.txt | head -n 10 > 50_audio.txt 
    python extract_feat.py -m 11 -x 12 -s -p extract -t 50_audio.txt
    c=`expr $c - 10`
done

if [ "$c" -gt 0 ]
then
    tail -n "$c" audio_files.txt > 50_audio.txt
    python extract_feat.py -m 11 -x 12 -s -p extract -t 50_audio.txt
fi
