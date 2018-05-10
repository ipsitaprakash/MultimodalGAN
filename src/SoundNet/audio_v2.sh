#!/bin/sh
s=33595 
while [ "$s" -gt 50 ] 
do
    tail -n "$s" urmp_audio_files.txt | head -n 50 > 50_2_audio.txt 
    python extract_feat.py -m 11 -x 12 -s -p extract -t 50_2_audio.txt
    s=`expr $s - 50`
done

if [ "$s" -gt 0 ]
then
    tail -n "$s" urmp_audio_files.txt > 50_2_audio.txt
    python extract_feat.py -m 11 -x 12 -s -p extract -t 50_2_audio.txt
fi
