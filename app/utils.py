#app/utils.py
import numpy as np
import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List, Tuple


def trim_zero_padding(sequence: np.ndarray) -> np.ndarray:
    non_zero_mask = np.any(sequence != 0, axis=1)
    return sequence[non_zero_mask]

def filter_short_intervals(intervals: list[tuple[int, int, str]], min_length: int = 5):
    return [(s, e, label) for s, e, label in intervals if (e - s) >= min_length]

def merge_gesture_intervals(intervals: list[tuple[int, int, str]], min_merge_gap: int = 0):
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    cs, ce, cl = sorted_intervals[0]
    for s, e, label in sorted_intervals[1:]:
        if label == cl and s <= ce + min_merge_gap:
            ce = max(ce, e)
        else:
            merged.append((cs, ce, cl))
            cs, ce, cl = s, e, label
    merged.append((cs, ce, cl))
    return merged

def sliding_window_gesture_detection(
        continuous_sequence: np.ndarray,
        encoder_model,
        gesture_hmms: dict[str, object],
        final_model,
        window_size: int = 20,
        step: int = 2,
        threshold_diff: float = 3.0,
        min_merge_gap: int = 5,
        min_interval_length: int = 5,
        debug: bool = False,
) -> List[Tuple[int, int, str]]:
    T = continuous_sequence.shape[0]
    if T < window_size:
        return []

    MAX_SEQ_LEN = encoder_model.input_shape[1]
    NUM_FEATURES = encoder_model.input_shape[2]
    gesture_names = list(gesture_hmms.keys())

    # Step 1: Collect all windows and pad
    n_windows = (T - window_size) // step + 1
    all_padded = []
    for w in range(n_windows):
        start = w * step
        end = start + window_size
        window = continuous_sequence[start:end, :]
        pad_len = MAX_SEQ_LEN - window.shape[0]
        if pad_len > 0:
            padded = np.vstack([window, np.zeros((pad_len, NUM_FEATURES), dtype=window.dtype)])
        else:
            padded = window[:MAX_SEQ_LEN, :]
        all_padded.append(padded)

    batched = np.stack(all_padded)  # shape: (n_windows, MAX_SEQ_LEN, NUM_FEATURES)

    # Step 2: Batch predict latent sequences
    latent_seqs = encoder_model.predict(batched, verbose=0)  # shape: (n_windows, MAX_SEQ_LEN, latent_dim)
    lengths = [MAX_SEQ_LEN] * n_windows

    # Step 3: Score using final model
    final_scores = [final_model.score(seq, [MAX_SEQ_LEN]) for seq in latent_seqs]

    # Step 4: Detect gestures
    detected = []
    for i, (latent_seq, f_ll) in enumerate(zip(latent_seqs, final_scores)):
        best_label = None
        max_diff = -np.inf
        scores = {}

        for name in gesture_names:
            g_ll = gesture_hmms[name].score(latent_seq, [MAX_SEQ_LEN])
            diff = g_ll - f_ll
            scores[name] = (g_ll, diff)
            if diff > max_diff:
                max_diff = diff
                best_label = name

        if debug:
            print(f"[DEBUG] w={i:03d} | f_ll={f_ll:.2f} | max_diff={max_diff:.2f} → {best_label}")
            for name in gesture_names:
                g_ll, diff = scores[name]
                print(f"    → {name:<10}: g_ll={g_ll:.2f}, diff={diff:.2f}")

        if max_diff >= threshold_diff:
            start = i * step
            end = start + window_size
            detected.append((start, end, best_label))

    # Step 5: Merge and filter results
    merged = merge_gesture_intervals(detected, min_merge_gap)
    final = filter_short_intervals(merged, min_interval_length)
    return final

def clean_json_sequence(
        sequence: list[list[float]],
        expected_length: int = None,
        max_seq_len: int = None
) -> np.ndarray:
    if expected_length is None and len(sequence) > 0:
        expected_length = len(sequence[0])

    cleaned = []
    for i, row in enumerate(sequence):
        if not isinstance(row, list):
            raise ValueError(f"Row {i} is not a list.")

        row = row[:expected_length] + [0.0] * max(0, expected_length - len(row))
        cleaned.append(row)

    arr = np.array(cleaned, dtype=np.float32)

    if max_seq_len is not None:
        arr = pad_sequences([arr], maxlen=max_seq_len, dtype='float32', padding='post', truncating='post')[0]

    return arr