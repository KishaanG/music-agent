import json
import numpy as np

from file_reading import getJSONFileFullPathname

class SongRating():
    def __init__(self, playlist_id):
        with open(getJSONFileFullPathname(playlist_id), "r", encoding="utf-8") as file:
            self._data = json.load(file)
        self._playlist_id = playlist_id
        self._ids = np.array(list(self._data["song_data"].keys()))
        np.random.shuffle(self._ids)
        self._ratings = {}
        self._confidence = 0.1569
        self._total_playlist_length = len(self._ids)
        self._current_song = None

    def allRatingsCompleted(self):
        return len(self._ids) <= 0

    def ratingCompleted(self, id):
        return id in self._ratings

    def _getNextID(self):
        if self.allRatingsCompleted():
            print(f"Ratings for playlist: {self._data["playlist_data"]["playlist_title"]} ({self._playlist_id}) are completed.")
            self._current_song = None
            return None
        
        if (
            self._current_song is not None and 
            not self.ratingCompleted(self._current_song)
        ):
            return self._current_song
        
        self._current_song = self._ids[0].item()
        self._ids = np.delete(self._ids, 0)
        return self._current_song

    def rate(self, like):
        if type(like) is not bool:
            raise TypeError(f"rateSong(like={like}): like must be a boolean")
        self._ratings[self._current_song] = like

    def getPlaylistData(self):
        return self._data["playlist_data"]

    def getConfidence(self):
        return f"{(self._confidence * 100):.2f}"

    def getTotalPlaylistLength(self):
        return self._total_playlist_length

class Comparison():
    def getID(p1_id, p2_id):
        return f"c_{p1_id}+{p2_id}"

    def __init__(self, p1_id, p2_id):
        self._p1_id = p1_id
        self._p2_id = p2_id
        self.id = Comparison.getID(self._p1_id, self._p2_id)
        self.sr1 = SongRating(self._p1_id)
        self.sr2 = SongRating(self._p2_id)
        self._sr = [self.sr1, self.sr2]
        self._current = None
        self._next()

    def done(self):
        return self.sr1.allRatingsCompleted() and self.sr2.allRatingsCompleted()

    def _whosNext(self):
        if self.sr1.allRatingsCompleted():
            return 1, 0
        if self.sr2.allRatingsCompleted():
            return 0, 1
        idx = 0 if (self.sr1.getConfidence() <= self.sr2.getConfidence()) else 1
        return idx, 1 - idx

    def _next(self):
        if self._current is not None:
            raise RuntimeError("Rate current song before getting next one.")
        idx_o, idx_r = self._whosNext()
        self._current = {
            "owner": {
                "idx": idx_o,
                "name": self._sr[idx_o].getPlaylistData()["owner_name"],
                "id": self._sr[idx_o].getPlaylistData()["owner_channel_id"]
            },
            "rater": {
                "idx": idx_r,
                "name": self._sr[idx_r].getPlaylistData()["owner_name"],
                "id": self._sr[idx_r].getPlaylistData()["owner_channel_id"]
            },
            "song": self._sr[idx_o]._getNextID()
        }

    def rate(self, like):
        if type(like) is not bool:
            raise TypeError(f"rateSong(like={like}): like must be a boolean")
        self._sr[self._current["owner"]["idx"]].rate(like)
        self._current = None
        self._next()

    def current(self):
        return self._current

def main():
    c = Comparison(p1_id="PLBr8POov-SoI", p2_id="PLnK3hfRa7cn1lWY4Qkwsf-x-dHoAand-v")
    print(c.current()["song"])
    c.rate(True)
    print(c.sr1._ratings)
    print(c.current()["song"])

if __name__ == "__main__":
    main()