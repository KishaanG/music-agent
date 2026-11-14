import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from yt_dlp import YoutubeDL
from yt_dlp.postprocessor.common import PostProcessor

import youtube_features as yf
from audio_features import AudioFeatures

# Defines maximum amount of threads that can be run concurrently
# Set to 4 for now to test. Will look into what an amount should be later.
MAX_WORKERS = 4

# Post Processor for YouTube Downloading: run() will happen after each video to get Audio Features
class GetAudioFeaturesPP(PostProcessor):
    def __init__(self, results, data, id, downloader=None):
        super().__init__(downloader)
        self.results = results
        self.data = data
        self.id = id

    def run(self, info):
        filepath = info.get('filepath')
        if filepath is None:
            return [], info

        self.to_screen(f'Getting Audio Features for "{self.results[self.id]["title"]} ({self.id})"')
        af = AudioFeatures(filepath)
        self.data[self.id] = af
        self.results[self.id].update(af.getFeatures())
        self.results[self.id]["features_found"] = True
        self.to_screen(f'Extracted Audio Features for "{self.results[self.id]["title"]} ({self.id})"')
        
        return [filepath], info # return list_of_files_to_delete, info_dict

class SongFeature():
    def __init__(self, playlist_url, ffmpeg_location, path='./'):
        self._results = yf.extractPlaylistFeatures(playlist=playlist_url)
        self._audio_data = {}
        self._ffmpeg_location = ffmpeg_location
        self._path = path

        # Runs SongFeature.download() concurrently for each video in playlist
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as tpe:
            futures = { 
                tpe.submit(
                    self.download, 
                    v["url"], 
                    k
                ): k for k,v in self._results.items()
            }
            for f in as_completed(futures):
                item = futures[f]
                try:
                    f.result()
                except Exception as e:
                    print(f"ERROR for {item}: {e}")

    def getFeatures(self):
        return self._results

    def displayPlotById(self, id):
        af = self._audio_data.get(id)

        if af is None:
            return

        af.plot(title=self._results[id]["title"])

    def download(self, url, id):
        if self._results[id].get("features_found"):
            return
        opts = {
            "format": "bestaudio/best",
            'outtmpl': f'{self._path}/audio_files/%(id)s.%(ext)s',
            'ffmpeg_location': self._ffmpeg_location,
            'keepvideo': False,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav"
            }]
        }
        with YoutubeDL(opts) as ydl:
            ydl.add_post_processor(GetAudioFeaturesPP(results=self._results, data=self._audio_data, id=id), when='after_move')
            ydl.download(url)

# -- FINDING ffmpeg LOCATION --
# If you do not have ffmpeg, enter: winget install ffmpeg
# Then, run: where ffmpeg
def main():
    sf = SongFeature(playlist_url=sys.argv[1], ffmpeg_location=sys.argv[2])
    with open('./data/test.json', 'w', encoding='utf-8') as file:
        json.dump(sf.getFeatures(), file, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()