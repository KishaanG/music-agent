import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from yt_dlp import YoutubeDL
from yt_dlp.postprocessor.common import PostProcessor

import youtube_features as yf
from audio_features import AudioFeatures

from file_reading import getJSONFileFullPathname, getAudioFolderPathname

__all__ = [ "SongFeature" ]

FFMPEG_LOCATION = os.getenv("FFMPEG_LOCATION")
MAX_WORKERS = os.cpu_count() / 4

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

        self.to_screen(f'Getting Audio Features for "{self.results["song_data"][self.id]["title"]} ({self.id})"')
        af = AudioFeatures(filepath)
        self.data[self.id] = af
        self.results["song_data"][self.id].update(af.getFeatures())
        self.results["song_data"][self.id]["features_found"] = True
        self.to_screen(f'Extracted Audio Features for "{self.results["song_data"][self.id]["title"]} ({self.id})"')
        
        return [filepath], info # return list_of_files_to_delete, info_dict

class SongFeature():
    def __init__(self, playlist_url):
        self._results, self._playlist_id = yf.extractPlaylistFeatures(playlist=playlist_url)
        self._audio_data = {}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as tpe:
            futures = { 
                tpe.submit(
                    self._download, 
                    v["url"], 
                    k
                ): k for k,v in self._results["song_data"].items()
            }
            for f in as_completed(futures):
                item = futures[f]
                try:
                    f.result()
                except Exception as e:
                    print(f"ERROR for {item}: {e!r}")

    def getFeatures(self):
        return self._results

    def getPlaylistId(self):
        return self._playlist_id

    def exportFeaturesToJSON(self):
        with open(getJSONFileFullPathname(self._playlist_id), "w", encoding="utf-8") as file:
            json.dump(self._results, file, indent=2, ensure_ascii=False)        

    def displayPlotById(self, id):
        af = self._audio_data.get(id)

        if af is None:
            return

        af.plot(title=self._results["song_data"][id]["title"])

    def _download(self, url, id):
        if self._results["song_data"][id].get("features_found"):
            return
        opts = {
            "format": "bestaudio/best",
            "outtmpl": f"{getAudioFolderPathname()}/%(id)s.%(ext)s",
            "ffmpeg_location": FFMPEG_LOCATION,
            "keepvideo": False,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav"
            }]
        }
        with YoutubeDL(opts) as ydl:
            ydl.add_post_processor(GetAudioFeaturesPP(results=self._results, data=self._audio_data, id=id), when='after_move')
            ydl.download(url)

def main():
    SongFeature(playlist_url=sys.argv[1]).exportFeaturesToJSON()

if __name__ == "__main__":
    main()  