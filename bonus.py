"""
===========================================================================
PART 3 - Bonus (SAM-M.EIC026 / ANS-MM0050)
===========================================================================
3a) Confidence level (0-100%) for each clip-song match.
3b) Overall success rate over all 22 clips.
3c) Improved recognition performance by trying different MFCC vector lengths
    and number of Mel filters, reporting which configuration works best.
3d) Real-time functionality: listens via microphone for ~10 seconds,
    then identifies the song.

How to run:
    python bonus.py              # runs 3a + 3b + 3c (offline tests)
    python bonus.py --realtime   # runs 3d (microphone identification)
===========================================================================
"""

import os
import sys
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct

import librosa

# Reuse the existing modules from the group
from project import fingerprint, N, N2, win
from identification import (
    build_database, identify_clip, cosine_similarity,
    expected_song, play_audio, DB_DIR, CLIPS_DIR
)

# ===========================================================================
# 3a) CONFIDENCE LEVEL
# ===========================================================================
# The cosine similarity score is in [-1, 1].
# We map it to [0%, 100%] using: confidence = (score + 1) / 2 * 100
# A score below 50% means low confidence (as required by the assignment).
# We also factor in the margin between the best and second-best match:
# a large margin means the system is more certain.

def confidence_level(best_score, second_score):
    """
    Returns a confidence percentage (0-100) based on:
    - The best cosine similarity score mapped to [0, 100]
    - A margin bonus: how much better the best match is vs the second best

    The formula is:
        base      = (best_score + 1) / 2 * 100   [maps [-1,1] -> [0,100]]
        margin    = (best_score - second_score) * 50  [extra certainty bonus]
        confidence = clamp(base + margin, 0, 100)
    """
    base = (best_score + 1) / 2 * 100
    margin_bonus = (best_score - second_score) * 50
    confidence = float(np.clip(base + margin_bonus, 0, 100))
    return confidence


# ===========================================================================
# 3b) OVERALL SUCCESS RATE
# ===========================================================================

def evaluate_all(db, label="Baseline"):
    """
    Runs identification on every clip in songs_clips/, computes:
    - per-clip result (correct / wrong)
    - confidence level (3a)
    - overall success rate (3b)

    Prints a formatted table and returns (correct, total, success_rate).
    """
    clips = sorted(f for f in os.listdir(CLIPS_DIR) if f.endswith(".wav"))
    total = len(clips)
    correct = 0

    print("=" * 90)
    print(f"  EVALUATION: {label}")
    print("=" * 90)
    print(f"{'CLIP':<42} {'BEST MATCH':<30} {'CONF':>6}  {'OK?'}")
    print("-" * 90)

    for clip in clips:
        best_song, best_score, ranking = identify_clip(
            os.path.join(CLIPS_DIR, clip), db
        )
        second_score = ranking[1][1] if len(ranking) > 1 else 0.0
        conf = confidence_level(best_score, second_score)
        is_correct = (best_song == expected_song(clip))
        correct += int(is_correct)

        short_match = best_song.replace("_30.wav", "")
        ok_marker = "✓" if is_correct else "✗"
        print(f"  {clip:<40} {short_match:<30} {conf:5.1f}%  {ok_marker}")

    success_rate = correct / total * 100 if total > 0 else 0
    print("=" * 90)
    print(f"  Result: {correct}/{total} correct  →  Success rate: {success_rate:.1f}%")
    print("=" * 90)
    return correct, total, success_rate


# ===========================================================================
# 3c) IMPROVE PERFORMANCE
# ===========================================================================
# We test different combinations of:
#   - n_mfcc  : number of MFCC coefficients kept after DCT (13, 20, 30)
#   - n_mels  : number of Mel filterbank bands (64, 128, 256)
# For each combination we rebuild the database and re-run the evaluation.
# The baseline uses n_mfcc=13, n_mels=128 (same as project.py).

def fingerprint_custom(inpfile, n_mfcc=13, n_mels=128):
    """
    Same pipeline as project.py's fingerprint() but with configurable
    n_mfcc and n_mels so we can experiment in 3c.
    """
    FS, data_sinal = wavfile.read(inpfile)
    data_sinal = data_sinal.astype(np.float32)
    data_sinal = data_sinal / (np.max(np.abs(data_sinal)) + 1e-10)
    if len(data_sinal.shape) > 1:
        data_sinal = np.mean(data_sinal, axis=1)

    mel_filterbank = librosa.filters.mel(sr=FS, n_fft=N, n_mels=n_mels)
    mfcc_list = []

    for i in range(0, len(data_sinal) - N, N2):
        frame = data_sinal[i:i + N]
        if len(frame) < N:
            break
        win_aux = frame * win
        magnitude_spectrum = np.abs(np.fft.rfft(win_aux))
        power_spectrum = magnitude_spectrum ** 2
        mel_energies = np.dot(mel_filterbank, power_spectrum)
        log_mel = np.log(mel_energies + 1e-10)
        mfcc = dct(log_mel, type=2, norm='ortho')[:n_mfcc]
        mfcc_list.append(mfcc)

    mfcc_list = np.array(mfcc_list)
    if len(mfcc_list) == 0:
        return np.zeros(n_mfcc)
    return np.mean(mfcc_list, axis=0)


def build_database_custom(n_mfcc=13, n_mels=128):
    db = {}
    for song in sorted(os.listdir(DB_DIR)):
        if not song.endswith(".wav"):
            continue
        fp = fingerprint_custom(os.path.join(DB_DIR, song), n_mfcc, n_mels)
        db[song] = fp
    return db


def identify_clip_custom(clip_path, db, n_mfcc=13, n_mels=128):
    fp_clip = fingerprint_custom(clip_path, n_mfcc, n_mels)
    scores = [(song, cosine_similarity(fp_song, fp_clip))
              for song, fp_song in db.items()]
    ranking = sorted(scores, key=lambda x: x[1], reverse=True)
    best_song, best_score = ranking[0]
    return best_song, best_score, ranking


def evaluate_custom(n_mfcc, n_mels, label=None):
    if label is None:
        label = f"n_mfcc={n_mfcc}, n_mels={n_mels}"
    db = build_database_custom(n_mfcc, n_mels)
    clips = sorted(f for f in os.listdir(CLIPS_DIR) if f.endswith(".wav"))
    total = len(clips)
    correct = 0
    for clip in clips:
        best_song, _, _ = identify_clip_custom(
            os.path.join(CLIPS_DIR, clip), db, n_mfcc, n_mels
        )
        if best_song == expected_song(clip):
            correct += 1
    rate = correct / total * 100 if total > 0 else 0
    print(f"  {label:<35}  {correct:>2}/{total}  ({rate:.1f}%)")
    return correct, total, rate


def run_parameter_sweep():
    """
    Tests different (n_mfcc, n_mels) combinations and reports which is best.
    """
    configs = [
        (13,  64),
        (13, 128),   # baseline
        (13, 256),
        (20,  64),
        (20, 128),
        (20, 256),
        (30,  64),
        (30, 128),
        (30, 256),
    ]

    print("\n" + "=" * 60)
    print("  3c) PARAMETER SWEEP")
    print("=" * 60)
    print(f"  {'Configuration':<35}  {'Result':>8}")
    print("-" * 60)

    results = []
    for n_mfcc, n_mels in configs:
        label = f"n_mfcc={n_mfcc}, n_mels={n_mels}"
        if n_mfcc == 13 and n_mels == 128:
            label += "  ← baseline"
        c, t, r = evaluate_custom(n_mfcc, n_mels, label)
        results.append((label, c, t, r))

    best = max(results, key=lambda x: x[3])
    print("=" * 60)
    print(f"  Best config: {best[0]}")
    print(f"  Best result: {best[1]}/{best[2]}  ({best[3]:.1f}%)")
    print("=" * 60)


# ===========================================================================
# 3d) REAL-TIME MICROPHONE IDENTIFICATION
# ===========================================================================

def realtime_identify(duration=10):
    """
    Records `duration` seconds from the microphone, extracts the fingerprint,
    and identifies the best matching song in the database.

    Requires: sounddevice  (pip install sounddevice)
    """
    try:
        import sounddevice as sd
    except ImportError:
        print("ERROR: sounddevice not installed.")
        print("  Run:  pip install sounddevice")
        return

    SAMPLE_RATE = 22050

    print(f"\n3d) REAL-TIME IDENTIFICATION")
    print(f"  Recording for {duration} seconds... (speak/play now)")
    audio = sd.rec(int(duration * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype='float32')
    sd.wait()
    print("  Recording done. Processing...")

    # Flatten to 1-D
    audio = audio.flatten()
    # Normalise
    audio = audio / (np.max(np.abs(audio)) + 1e-10)

    # Extract fingerprint using the same pipeline as project.py
    mel_filterbank = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N, n_mels=128)
    mfcc_list = []
    for i in range(0, len(audio) - N, N2):
        frame = audio[i:i + N]
        if len(frame) < N:
            break
        win_aux = frame * win
        magnitude_spectrum = np.abs(np.fft.rfft(win_aux))
        power_spectrum = magnitude_spectrum ** 2
        mel_energies = np.dot(mel_filterbank, power_spectrum)
        log_mel = np.log(mel_energies + 1e-10)
        mfcc = dct(log_mel, type=2, norm='ortho')[:13]
        mfcc_list.append(mfcc)

    if len(mfcc_list) == 0:
        print("  ERROR: No audio frames captured. Check microphone.")
        return

    fp_clip = np.mean(np.array(mfcc_list), axis=0)

    # Load database and match
    print("  Building fingerprint database...")
    db = build_database()

    scores = [(song, cosine_similarity(fp_song, fp_clip))
              for song, fp_song in db.items()]
    ranking = sorted(scores, key=lambda x: x[1], reverse=True)
    best_song, best_score = ranking[0]
    second_score = ranking[1][1] if len(ranking) > 1 else 0.0
    conf = confidence_level(best_score, second_score)

    print(f"\n  >>> Best match : {best_song}")
    print(f"  >>> Similarity : {best_score:.4f}")
    print(f"  >>> Confidence : {conf:.1f}%")

    if conf < 50:
        print("  (Low confidence — the song may not be in the database)")

    print("\n  Playing the identified song...")
    play_audio(os.path.join(DB_DIR, best_song))


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":

    if "--realtime" in sys.argv:
        # 3d) real-time microphone mode
        duration = 10
        for arg in sys.argv[1:]:
            if arg.isdigit():
                duration = int(arg)
        realtime_identify(duration=duration)

    else:
        # 3a + 3b: baseline evaluation with confidence levels
        print("\nBuilding baseline fingerprint database...")
        db = build_database()
        print(f"{len(db)} songs loaded.\n")

        evaluate_all(db, label="Baseline  (n_mfcc=13, n_mels=128)")

        # 3c: parameter sweep
        run_parameter_sweep()
