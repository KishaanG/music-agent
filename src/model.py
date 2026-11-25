import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

def merge_playlists(songsA, songsB):
	result = {}
	for id, song in songsA.items():
		result[id] = {**song, "owner": "A"}
	for id, song in songsB.items():
		owner = "B"
		if id in result: owner = "Both"
		result[id] = {**song, "owner": owner}
	return result

# Audio features from AudioFeatures.getFeatures(), all numeric
AUDIO_COLUMNS = [
	"tempo",
	"beat_strength",
	"integrated_loudness",
	"loudness_range",
	"main_frequency",
	"main_frequency_range",
	"time_to_peak",
	"volume_mean",
	"volume_std",
	"bass_volume_mean",
	"bass_volume_std",
	"bass_difference",
]


def getYear(song):
	date = song.get("release_date")
	if date is None:
		date = song.get("upload_date")
	if date is None:
		return np.nan
	return float(date[:4]) # return just the year

def getViews(song):
	views = song.get("view_count")
	if views is None:
		return np.nan
	else:
		return np.log1p(views)  # log so bigger view counts don't dominate

def numericFeatures(song):
	row = []
	for column in AUDIO_COLUMNS:
		value = song.get(column)
		if value is None:
			row.append(np.nan)
		else:
			row.append(float(value))

	row.append(getViews(song))
	row.append(getYear(song))
	return row


def getAllGenres(songs):
	all_genres = set()
	for song in songs.values():
		all_genres.update(song.get("genres", []))
	return sorted(all_genres) # return all the genres represented in all the songs provided


def genreFeatures(song, all_genres):
	song_genres = song.get("genres", [])
	row = []
	for genre in all_genres:
		if genre in song_genres:
			row.append(1.0)
		else:
			row.append(0.0)
	return row # return 1 if genre is in the song, 0 if not


def buildFeatureMatrix(songs):
	if not songs: raise ValueError("No songs")
	
	song_ids = list(songs.keys())
	all_genres = getAllGenres(songs)

	number_rows = []
	genre_rows = []
	for id in song_ids:
		number_rows.append(numericFeatures(songs[id])) # 1 list per song
		genre_rows.append(genreFeatures(songs[id], all_genres))
	numbers = np.array(number_rows) # lists of lists (vertical array)
	genres = np.array(genre_rows)
	
	mean = np.nanmean(numbers, axis=0)
	std = np.nanstd(numbers, axis=0)
	std = np.where(std == 0, 1, std) # avoid divide by 0 error
	numbers = np.nan_to_num((numbers - mean) / std)
	
	bias = np.ones((len(song_ids), 1))

	X = np.hstack([bias, numbers, genres])
	feature_names = ["bias"] + AUDIO_COLUMNS + ["log_view_count", "release_year"] + all_genres
	return X, song_ids, feature_names

class BayesianLogisticRegression:
	def __init__(self, _priorMean, _priorCov):
		_priorMean = np.array(_priorMean, dtype=float)
		_priorCov = np.array(_priorCov, dtype=float)
		n_features = len(_priorMean)
		if _priorMean.ndim != 1 or _priorCov.shape != (n_features, n_features):
			raise ValueError(f"priorCov must have shape ({n_features}, {n_features}) to match priorMean")
		if not np.allclose(_priorCov, _priorCov.T):
			raise ValueError("priorCov must be symmetric")
		try:
			np.linalg.cholesky(_priorCov)  # errors if any variance is <= 0
		except np.linalg.LinAlgError:
			raise ValueError("priorCov must be positive definite (all variances > 0)") from None
		self.priorCov = _priorCov
		self.priorMean = _priorMean
		self.priorPrecision = np.linalg.inv(_priorCov)  # like 1 / variance: big = confident
		self.postMean = _priorMean.copy()
		self.postCov = _priorCov.copy()
		self.X_seen = np.empty((0,len(_priorMean)))
		self.y_seen = np.empty(0)

	def loss(self, w):
		z = self.X_seen @ w
		negLogLikelihood = np.sum(np.logaddexp(0, z) - self.y_seen * z)  # how badly w explains the ratings
		diff = w - self.priorMean
		negLogPrior = 0.5 * diff @ self.priorPrecision @ diff  # penalty for moving away from the prior
		return negLogLikelihood + negLogPrior

	def gradient(self, w):
		probs = expit(self.X_seen @ w)
		return self.X_seen.T @ (probs - self.y_seen) + self.priorPrecision @ (w - self.priorMean)

	def hessian(self, w):
		probs = expit(self.X_seen @ w)
		curvature = probs * (1 - probs)
		return (self.X_seen.T * curvature) @ self.X_seen + self.priorPrecision  # how sharply the loss curves

	def validateRatings(self, X, y):
		X = np.atleast_2d(np.asarray(X, dtype=float))
		y = np.atleast_1d(np.asarray(y, dtype=float))
		n_features = len(self.priorMean)
		if X.ndim != 2 or X.shape[1] != n_features:
			raise ValueError(f"X must have shape (n_songs, {n_features}), got {X.shape}")
		if y.ndim != 1 or len(y) != len(X):
			raise ValueError(f"y must have one rating per song: {len(X)} songs, {y.shape} ratings")
		if not np.all(np.isfinite(X)):
			raise ValueError("X contains NaN or inf; fill missing features first")
		if not np.all(np.isin(y, [0.0, 1.0])):
			raise ValueError("y must be 0 (dislike) or 1 (like)")
		return X, y

	def update(self, X, y):
		X, y = self.validateRatings(X, y)
		old_X, old_y = self.X_seen, self.y_seen
		self.X_seen = np.vstack([old_X, X])
		self.y_seen = np.concatenate([old_y, y])

		try:
			result = minimize(self.loss, self.postMean, jac=self.gradient, hess=self.hessian, method="trust-exact")  # finds the w with the lowest loss
			if not result.success:
				raise RuntimeError(f"posterior fit did not converge: {result.message}")
			postCov = np.linalg.inv(self.hessian(result.x))  # sharp curve = small uncertainty
		except Exception:
			self.X_seen, self.y_seen = old_X, old_y  # undo, so a failed update changes nothing
			raise

		self.postMean = result.x
		self.postCov = (postCov + postCov.T) / 2  # rounding can make it slightly unsymmetric

	def predict(self, X):
		X = np.atleast_2d(np.asarray(X, dtype=float))
		logitMean = X @ self.postMean
		logitVar = np.einsum("ij,jk,ik->i", X, self.postCov, X)  # x^T postCov x for every song
		logitVar = np.maximum(logitVar, 0)
		probs = expit(logitMean / np.sqrt(1 + np.pi * logitVar / 8))  # more uncertainty pulls p toward 0.5
		return probs, np.sqrt(logitVar)

	def sampleWeights(self, rng=None):
		if rng is None:
			rng = np.random.default_rng()
		return rng.multivariate_normal(self.postMean, self.postCov)  # one random guess at the user's taste
