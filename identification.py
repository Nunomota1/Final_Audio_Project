"""
===========================================================================
 PART 2 - Best match identification (SAM-M.EIC026 / ANS-MM0050)
===========================================================================
This module implements step 2 of the assignment:
    2_a)  read an audio clip and extract its fingerprint (same procedure as
          Part 1);
    2_b)  compare the clip fingerprint against all pre-computed fingerprints
          using cosine similarity (Pearson) and pick the highest as the best
          match;
    2_c)  play the audio clip and the identified song for human verification.

It reuses the fingerprint() function from Part 1 (mean MFCC extraction),
exactly as required: "the processing procedure is the same as that used to
extract the fingerprints of the reference songs".

How to run:
    python identification.py            # identify all clips (quick test)
    python identification.py <clip.wav> # identify one clip and play the audio
===========================================================================
"""

import os
import sys
import numpy as np

# Reuse the fingerprint extraction function written in Part 1.

from project import fingerprint

DB_DIR = "songs_database"
CLIPS_DIR = "songs_clips"


# ---------------------------------------------------------------------------
# 2_b_i)  Cosine similarity / Pearson correlation between 2 fingerprints
# ---------------------------------------------------------------------------
def cosine_similarity(fp_song, fp_clip):
    """
    Computes the similarity between a song fingerprint (fp_song) and the clip
    fingerprint (fp_clip), following the formula given in the assignment:

        CS = sum(a[i]*b[i]) / ( sqrt(sum(a[i]^2)) * sqrt(sum(b[i]^2)) )

    This expression is equivalent to the Pearson coefficient / corrcoef when
    the vectors are centered. We implement the given formula directly so the
    computation is explicit.
    """
    fp_song = np.asarray(fp_song, dtype=np.float64)
    fp_clip = np.asarray(fp_clip, dtype=np.float64)

    numerator = np.dot(fp_song, fp_clip)
    denominator = np.linalg.norm(fp_song) * np.linalg.norm(fp_clip)

    if denominator == 0:
        return 0.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# Pre-compute the fingerprints of all reference songs (30s)
# ---------------------------------------------------------------------------
def build_database():
    """
    Goes through songs_database/ and computes the fingerprint (mean MFCC) of
    each reference song. Returns a dictionary {filename: fingerprint}.
    """
    db = {}
    for song in sorted(os.listdir(DB_DIR)):
        if not song.endswith(".wav"):
            continue
        fp, _, _, _ = fingerprint(os.path.join(DB_DIR, song))
        db[song] = fp
    return db


# ---------------------------------------------------------------------------
# 2_a) + 2_b)  Identify ONE clip: extract fingerprint and search best match
# ---------------------------------------------------------------------------
def identify_clip(clip_path, db):
    """
    2_a) reads the clip and extracts its fingerprint (same procedure as Part 1);
    2_b_i) compares it against all pre-computed fingerprints;
    2_b_ii) selects the highest similarity -> best match.

    Returns: (best_song, best_score, ranking)
      ranking is the list [(song, score), ...] sorted from best to worst.
    """
    fp_clip, _, _, _ = fingerprint(clip_path)

    scores = []
    for song, fp_song in db.items():
        scores.append((song, cosine_similarity(fp_song, fp_clip)))

    # 2_b_ii) sort by descending similarity; the first one is the best match
    ranking = sorted(scores, key=lambda x: x[1], reverse=True)
    best_song, best_score = ranking[0]

    return best_song, best_score, ranking


# ---------------------------------------------------------------------------
# Ground truth: a clip named "X_10[_a].wav" corresponds to song "X_30.wav"
# (used only to check correctness in the quick test below)
# ---------------------------------------------------------------------------
def expected_song(clip_name):
    """
    Derives the expected song from the clip name.
    e.g. 'Aug2024_RHCP_Track16_10_b.wav' -> 'Aug2024_RHCP_Track16_30.wav'
    """
    base = clip_name[:-4]  # drop ".wav"
    if len(base) >= 2 and base[-2] == "_" and base[-1] in "abcdefgh":
        base = base[:-2]
    if base.endswith("_10"):
        base = base[:-3] + "_30"
    return base + ".wav"


# ---------------------------------------------------------------------------
# 2_c)  Play the clip and the identified song 
# ---------------------------------------------------------------------------
def play_audio(path):
    """
    Plays a .wav file. Uses sounddevice if available, otherwise falls back to
    the operating system's audio player. If nothing is available, it just
    warns (it never crashes).
    """
    try:
        import soundfile as sf
        import sounddevice as sd
        signal_data, fs = sf.read(path)
        sd.play(signal_data, fs)
        sd.wait()
        return
    except Exception:
        pass

    try:
        from scipy.io import wavfile
        import sounddevice as sd
        fs, signal_data = wavfile.read(path)
        sd.play(signal_data, fs)
        sd.wait()
        return
    except Exception:
        pass

    import platform, subprocess
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.run(["afplay", path])
        elif system == "Windows":
            os.startfile(path)  
        else:
            subprocess.run(["aplay", path])
    except Exception:
        print(f"   (could not play automatically: {path})")


# ---------------------------------------------------------------------------
# Quick test: identify all clips at once and report how many are correct
# ---------------------------------------------------------------------------
def test_all(db):
    clips = sorted(f for f in os.listdir(CLIPS_DIR) if f.endswith(".wav"))
    total = len(clips)
    correct = 0

    print("=" * 74)
    print(f"{'CLIP':<46}{'BEST MATCH':<24} OK")
    print("=" * 74)

    for clip in clips:
        best_song, _, _ = identify_clip(os.path.join(CLIPS_DIR, clip), db)
        is_correct = (best_song == expected_song(clip))
        correct += int(is_correct)
        short = best_song.replace("_30.wav", "")
        print(f"{clip:<46}{short:<24} {'OK' if is_correct else 'X'}")

    print("=" * 74)
    print(f"Correct: {correct}/{total}")
    print("=" * 74)


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def main():
    print("Building the fingerprint database of the reference songs...")
    db = build_database()
    print(f"{len(db)} songs in the database.\n")

    # identify a single clip passed as argument (plays the audio - 2_c)
    if len(sys.argv) > 1:
        clip_path = sys.argv[1]
        if not os.path.isabs(clip_path) and not os.path.exists(clip_path):
            clip_path = os.path.join(CLIPS_DIR, clip_path)

        best_song, best_score, ranking = identify_clip(clip_path, db)

        print(f"Clip:        {os.path.basename(clip_path)}")
        print(f"Best match:  {best_song}")
        print(f"Similarity:  {best_score:.4f}")
        print("\nTop 3 candidates:")
        for song, score in ranking[:3]:
            print(f"   {score:.4f}  {song}")

        # 2_c) play clip + identified song
        print("\nPlaying the clip and the identified song...")
        play_audio(clip_path)
        play_audio(os.path.join(DB_DIR, best_song))
        return

    #  quick test over all clips
    test_all(db)


if __name__ == "__main__":
    main()