# music-agent

## Get All Features from Playlist
py .\extract_features.py '<playlist url>' '<path to ffmpeg.exe>'

## Add this function in extract_features -> main() to view Plots for a song
<SongFeature>.displayPlotById("<video id>")

## -- FINDING ffmpeg LOCATION --
1. Open Command Prompt
2. If you do not have ffmpeg, enter: winget install ffmpeg
3. Then enter: where ffmpeg