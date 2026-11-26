from flask import Flask, render_template, request, redirect, url_for

from extract_features import SongFeature
from file_reading import playlistAlreadyAnalyzed, getFilenames
from song_rating import Comparison

app = Flask(__name__)

YOUTUBE_URL_PREFIX = "https://www.youtube.com/playlist?list="

def renderIndex(**kwargs):
    return render_template("index.html", files=list(enumerate(getFilenames())), **kwargs)

def getComparisonFromPlaylistIDs(p1, p2):
    comp_id = Comparison.getID(p1, p2)
    if comp_id not in comparisons:
        comp = Comparison(p1, p2)
        comparisons[comp_id] = comp
    return comp_id

@app.route("/", methods=["GET"])
def index():
    return renderIndex()

@app.route("/", methods=["POST"])
def indexPOST():
    def processURL(url):
        if not url.startswith(YOUTUBE_URL_PREFIX):
            return renderIndex(error="Invalid YouTube Playlist URL\n")
    
        id = url.removeprefix(YOUTUBE_URL_PREFIX).split("&")[0]
        if playlistAlreadyAnalyzed(id):
            return renderIndex(error=f"Playlist with id ({id}) has already been analyzed.")
        
        try:
            sf = SongFeature(url)
            sf.exportFeaturesToJSON()
            return renderIndex(complete=True)
        except Exception as e:
            return renderIndex(error=e)

    def processPlaylists(playlists):
        if len(playlists) != 2:
            return renderIndex(error=f"You must choose exactly 2 playlists. You chose {len(playlists)}.")
        return redirect(url_for("compare", comp_id=getComparisonFromPlaylistIDs(playlists[0], playlists[1])))

    url = request.form.get("url")
    if url is not None:
        return processURL(url.strip())
    selected = request.form.getlist("selected")
    if selected is not None:
        return processPlaylists(selected)
    return renderIndex(error="No playlist uploaded or selected.")

comparisons = {}

@app.route("/compare/<comp_id>", methods=["GET"])
def compare(comp_id):
    if comp_id not in comparisons:
        return renderIndex(error="Comparison not found. Please select two playlists.")
    comp = comparisons[comp_id]
    print(comp.current())
    return render_template("compare.html", comp=comp)

@app.route("/compare/<comp_id>/rate", methods=["POST"])
def rate(comp_id):
    comp = comparisons.get(comp_id)
    if comp is None:
        return renderIndex(error="Comparison not found. Please select two playlists.")
    like = request.form.get("like") == "true"
    try:
        comp.rate(like)
    except Exception as e:
        renderIndex(error=e)
    return redirect(url_for("compare", comp_id=comp_id))