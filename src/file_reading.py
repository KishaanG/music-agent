import os

__all__ = [
    "getJSONFileFullPathname",
    "getIDFromFilename",
    "getFilenames",
    "playlistAlreadyAnalyzed",
    "getAudioFolderPathname"
]

TEMPORARY_AUDIO_FOLDER = "./audio_files"
FEATURE_DATA_FOLDER = "./feature_data"
FILENAME_PREFIX = "playlist_features_"
JSON_EXTENSION = ".json"

def getJSONFileFullPathname(id):
    return os.path.join(FEATURE_DATA_FOLDER, f"{FILENAME_PREFIX}{id}{JSON_EXTENSION}")

def getIDFromFilename(filename):
    return filename.removeprefix(FILENAME_PREFIX).removesuffix(JSON_EXTENSION)

def getFilenames():
    return [ getIDFromFilename(f) for f in os.listdir(FEATURE_DATA_FOLDER) if os.path.isfile(os.path.join(FEATURE_DATA_FOLDER, f)) ]

def playlistAlreadyAnalyzed(id):
    return os.path.isfile(getJSONFileFullPathname(id))

def getAudioFolderPathname():
    return TEMPORARY_AUDIO_FOLDER