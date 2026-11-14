# music-agent

## Get All Features from Playlist
py .\extract_features.py '\<playlist url\>' '\<path to ffmpeg.exe\>'

Audio will be created, THEN DELETED in ./audio_files (The folder will remain).
Feature data will be created as test.json in ./data.

## Add this function in extract_features -> main() to view Plots for a song
\<SongFeature\>.displayPlotById("\<video id\>")

## FINDING ffmpeg LOCATION
1. Open Command Prompt
2. If you do not have ffmpeg, enter: winget install ffmpeg
3. Then enter: where ffmpeg
