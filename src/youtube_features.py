import json
import os
import sys
from urllib.parse import unquote

import requests
from dotenv import load_dotenv

# Reads YT_API_KEY from the .env file in the project root
load_dotenv()
API_KEY = os.getenv("YT_API_KEY")

API_BASE_URL = "https://www.googleapis.com/youtube/v3"

# The API returns at most 50 results per page and accepts at most 50 ids per call
MAX_RESULTS = 50

# YouTube's category id for "Music"
MUSIC_CATEGORY_ID = "10"

# Used to turn durations like "PT3M33S" into seconds
SECONDS_PER_UNIT = {"D": 86400, "H": 3600, "M": 60, "S": 1}


def apiGet(endpoint, params):
    response = requests.get(f"{API_BASE_URL}/{endpoint}", params=params, headers={"X-goog-api-key": API_KEY}, timeout=15)
    if response.status_code != 200:
        raise RuntimeError(f"YouTube API {endpoint} failed: {response.text}")
    return response.json()


def getPlaylistItems(playlist_id):
    params = {"part": "snippet,contentDetails", "playlistId": playlist_id, "maxResults": MAX_RESULTS}
    items = []
    while True:
        page = apiGet("playlistItems", params)
        items.extend(page["items"])
        if "nextPageToken" not in page:
            return items
        params["pageToken"] = page["nextPageToken"]


def getByIds(endpoint, ids, parts):
    results = {}
    for start in range(0, len(ids), MAX_RESULTS):
        batch = ids[start:start + MAX_RESULTS]
        page = apiGet(endpoint, {"part": parts, "id": ",".join(batch)})
        for item in page["items"]:
            results[item["id"]] = item
    return results


def getPlaylistId(playlist):
    if "list=" in playlist:
        return playlist.split("list=")[1].split("&")[0]
    return playlist


def toInt(value):
    if value is None:
        return None
    return int(value)


def getDurationSeconds(duration):
    if not duration:
        return None
    total_seconds = 0
    number = ""
    for char in duration:
        if char.isdigit():
            number += char
        elif char in SECONDS_PER_UNIT:
            total_seconds += int(number) * SECONDS_PER_UNIT[char]
            number = ""
    return total_seconds


def getGenres(topic_urls):
    genres = []
    for url in topic_urls:
        name = unquote(url.split("/wiki/")[-1]).replace("_", " ")
        if name != "Music":
            genres.append(name)
    return genres


def getReleaseDate(description):
    for line in description.splitlines():
        if line.startswith("Released on:"):
            return line.removeprefix("Released on:").strip()
    return None


def getArtist(title, channel_title, description):
    if description.startswith("Provided to YouTube by"):
        for line in description.splitlines():
            if " · " in line:
                artists = line.split(" · ")[1:]
                return ", ".join(artists), "description"

    if channel_title.endswith(" - Topic"):
        return channel_title.removesuffix(" - Topic"), "topic_channel"

    for dash in [" - ", " – ", " — "]:
        if dash in title:
            return title.split(dash)[0].strip(), "title"

    return channel_title.removesuffix("VEVO").strip(), "channel"


def getVideoFeatures(video, channel):
    snippet = video["snippet"]
    stats = video.get("statistics", {})
    details = video.get("contentDetails", {})
    topics = video.get("topicDetails", {})
    channel_stats = channel.get("statistics", {})
    channel_topics = channel.get("topicDetails", {})

    title = snippet["title"]
    channel_title = snippet["channelTitle"]
    description = snippet.get("description", "")
    artist, artist_source = getArtist(title, channel_title, description)

    subscriber_count = toInt(channel_stats.get("subscriberCount"))
    if channel_stats.get("hiddenSubscriberCount"):
        subscriber_count = None

    video_data = {
        "video_id": video["id"],
        "url": "https://www.youtube.com/watch?v=" + video["id"],
        "title": title,
        "artist": artist,
        "artist_source": artist_source,
        "channel_id": snippet["channelId"],
        "channel_title": channel_title,
        "genres": getGenres(topics.get("topicCategories", [])),
        "channel_genres": getGenres(channel_topics.get("topicCategories", [])),
        "tags": snippet.get("tags", []),
        "category_id": snippet.get("categoryId"),
        "is_music_category": snippet.get("categoryId") == MUSIC_CATEGORY_ID,
        # Like and comment counts are missing when the uploader hides them
        "view_count": toInt(stats.get("viewCount")),
        "like_count": toInt(stats.get("likeCount")),
        "comment_count": toInt(stats.get("commentCount")),
        "duration_seconds": getDurationSeconds(details.get("duration")),
        "upload_date": snippet["publishedAt"],
        "release_date": getReleaseDate(description),
        "language": snippet.get("defaultAudioLanguage") or snippet.get("defaultLanguage"),
        "licensed_content": details.get("licensedContent"),
        "channel_subscriber_count": subscriber_count,
        "channel_view_count": toInt(channel_stats.get("viewCount")),
        "channel_video_count": toInt(channel_stats.get("videoCount")),
    }
    return video_data


def extractVideoFeatures(video_ids):
    videos = getByIds("videos", video_ids, "snippet,contentDetails,statistics,topicDetails")

    channel_ids = list(set(video["snippet"]["channelId"] for video in videos.values()))
    channels = getByIds("channels", channel_ids, "statistics,topicDetails")

    features = {}
    for video_id in video_ids:
        if video_id in videos:
            video = videos[video_id]
            channel = channels.get(video["snippet"]["channelId"], {})
            features[video_id] = getVideoFeatures(video, channel)
    return features


def extractPlaylistFeatures(playlist):
    items = getPlaylistItems(getPlaylistId(playlist))

    added_dates = {}
    for item in items:
        video_id = item["contentDetails"]["videoId"]
        if video_id not in added_dates:
            added_dates[video_id] = item["snippet"]["publishedAt"]

    video_ids = list(added_dates.keys())

    features = extractVideoFeatures(video_ids)
    for video_id in features:
        features[video_id]["added_to_playlist_at"] = added_dates[video_id]
    return features


if __name__ == "__main__":
    playlist = sys.argv[1]
    output_path = sys.argv[2]

    features = extractPlaylistFeatures(playlist)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(features, file, indent=2, ensure_ascii=False)
    print(f"Saved features for {len(features)} videos to {output_path}")
