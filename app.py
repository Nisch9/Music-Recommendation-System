import re
import json
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from Training import Spotify_Recommendation

# ── Load data ────────────────────────────────────────────────────────────────
# IMPORTANT: never add columns to `data` — Training.py iterates every column
# with float() and will crash on any string column you add.
data = pd.read_csv('Dataset/data.csv')

# Separate, lightweight lookup dataframe — safe to mutate
_lookup = data[['name', 'artists', 'energy', 'valence', 'danceability',
                'acousticness', 'instrumentalness', 'tempo']].copy()
_lookup['_name_lower'] = _lookup['name'].str.lower().fillna('')

app = Flask(__name__)

# ── Autocomplete lists ───────────────────────────────────────────────────────
song_names = sorted(data['name'].dropna().unique().tolist())
artist_names = set()
for _a in data['artists'].dropna().unique():
    try:
        parsed = eval(_a) if isinstance(_a, str) else []
        names  = parsed if isinstance(parsed, list) else [parsed]
        for n in names:
            artist_names.add(str(n).strip())
    except Exception:
        pass
artist_names = sorted(artist_names)

# ── Analytics ────────────────────────────────────────────────────────────────
total_songs        = len(data)
most_popular       = data.loc[data['popularity'].idxmax()]
most_popular_song  = most_popular['name']
most_popular_artist= most_popular['artists']
most_popular_count = most_popular['popularity']

top_songs    = data[['name','popularity']].sort_values('popularity', ascending=False).head(10)
chart_labels = top_songs['name'].tolist()
chart_values = top_songs['popularity'].tolist()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_artist_key(artist_str: str) -> str:
    """Parse artist strings like \"['Bad Bunny', 'J Balvin']\" → 'bad bunny'."""
    try:
        parsed = eval(artist_str)
        if isinstance(parsed, list) and parsed:
            return parsed[0].strip().lower()
    except Exception:
        pass
    return re.sub(r"[\[\]\"']", '', artist_str).split(',')[0].strip().lower()


def _norm_tempo(bpm: float) -> float:
    return max(0.0, min(1.0, (bpm - 50) / 160))


def _row_to_features(row) -> dict:
    return {
        'energy':          round(float(row['energy']),           4),
        'valence':         round(float(row['valence']),          4),
        'danceability':    round(float(row['danceability']),     4),
        'acousticness':    round(float(row['acousticness']),     4),
        'instrumentalness':round(float(row['instrumentalness']), 4),
        'tempo_norm':      round(_norm_tempo(float(row['tempo'])), 4),
    }


def get_audio_features(song_name: str, artist_str: str) -> dict:
    """
    4-strategy lookup against _lookup (NOT data) so Training.py stays clean.
    """
    name_lower = song_name.strip().lower()
    artist_key = _extract_artist_key(artist_str)

    # Strategy 1 — exact name + artist
    exact = _lookup[_lookup['_name_lower'] == name_lower]
    if not exact.empty and artist_key:
        wa = exact[exact['artists'].str.lower().str.contains(
            artist_key[:20], na=False, regex=False)]
        if not wa.empty:
            return _row_to_features(wa.iloc[0])

    # Strategy 2 — exact name only
    if not exact.empty:
        return _row_to_features(exact.iloc[0])

    # Strategy 3 — partial name
    try:
        partial = _lookup[_lookup['_name_lower'].str.contains(
            re.escape(name_lower[:25]), na=False, regex=True)]
        if not partial.empty:
            if artist_key:
                wa = partial[partial['artists'].str.lower().str.contains(
                    artist_key[:20], na=False, regex=False)]
                if not wa.empty:
                    return _row_to_features(wa.iloc[0])
            return _row_to_features(partial.iloc[0])
    except Exception:
        pass

    # Strategy 4 — artist only
    if artist_key:
        by_artist = _lookup[_lookup['artists'].str.lower().str.contains(
            artist_key[:20], na=False, regex=False)]
        if not by_artist.empty:
            return _row_to_features(by_artist.iloc[0])

    # Fallback — dataset medians
    return {
        'energy':          round(float(_lookup['energy'].median()),           4),
        'valence':         round(float(_lookup['valence'].median()),          4),
        'danceability':    round(float(_lookup['danceability'].median()),     4),
        'acousticness':    round(float(_lookup['acousticness'].median()),     4),
        'instrumentalness':round(float(_lookup['instrumentalness'].median()), 4),
        'tempo_norm':      round(_norm_tempo(float(_lookup['tempo'].median())), 4),
    }


def classify_moods(f: dict) -> list:
    e, v, d, i, t = f['energy'], f['valence'], f['danceability'], f['instrumentalness'], f['tempo_norm']
    moods = []
    if e >= 0.55 or (t >= 0.55 and d >= 0.55):     moods.append('energetic')
    if e <= 0.60:                                    moods.append('chill')
    if v >= 0.50:                                    moods.append('happy')
    if v <= 0.45:                                    moods.append('sad')
    if i >= 0.05 or (e <= 0.75 and d <= 0.70):      moods.append('focus')
    moods.append('all')
    return moods


def _enrich(raw_recs: list) -> list:
    result = []
    for artist, name in raw_recs:
        features = get_audio_features(name, artist)
        moods    = classify_moods(features)
        result.append({
            'artist': artist, 'name': name,
            'features': features, 'moods': moods,
            'moods_json': json.dumps(moods),
        })
    return result


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations, error = None, None

    if request.method == 'POST':
        song_name   = request.form.get('song_name',   '').strip()
        artist_name = request.form.get('artist_name', '').strip()
        try:    num_recs        = int(request.form.get('num_recs', 5))
        except: num_recs        = 5
        try:    num_recs_artist = int(request.form.get('num_recs_artist', 5))
        except: num_recs_artist = 5

        if song_name:
            try:
                # data is clean — Training.py will work fine
                rec      = Spotify_Recommendation(data)
                output   = rec.recommend(song_name, num_recs)
                raw      = output.values.tolist()   # [[artist, name], ...]
                recommendations = _enrich(raw)
            except Exception as ex:
                error = f"Error: {ex}"

        elif artist_name:
            try:
                mask = data['artists'].str.lower().str.contains(artist_name.lower(), na=False)
                raw  = data[mask][['artists','name']].head(num_recs_artist).values.tolist()
                recommendations = _enrich(raw)
                if not recommendations:
                    error = f"No songs found for artist '{artist_name}'."
            except Exception as ex:
                error = f"Error: {ex}"
        else:
            error = "Please enter a song or artist name."

    return render_template(
        'index.html',
        recommendations=recommendations, error=error,
        song_names=song_names, artist_names=artist_names,
        total_songs=total_songs,
        most_popular_song=most_popular_song,
        most_popular_artist=most_popular_artist,
        most_popular_count=most_popular_count,
        chart_labels=chart_labels, chart_values=chart_values,
    )


@app.route('/debug_features')
def debug_features():
    song, artist = request.args.get('song',''), request.args.get('artist','')
    feat  = get_audio_features(song, artist)
    moods = classify_moods(feat)
    return jsonify({'song': song, 'artist': artist, 'features': feat, 'moods': moods})


@app.route('/autocomplete')
def autocomplete():
    q = request.args.get('q','').lower()
    return jsonify([n for n in song_names if q in n.lower()][:10])


@app.route('/autocomplete_artist')
def autocomplete_artist():
    q = request.args.get('q','').lower()
    return jsonify([n for n in artist_names if q in n.lower()][:10])


if __name__ == '__main__':
    app.run(debug=True)