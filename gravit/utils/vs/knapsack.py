import numpy as np


def fill_knapsack(final_len, scores, lens):
    """
    Given a set of segments, each with a length and an importance value,
    this method determines which segments to include in a final summary
    so that total length is less than or equal to final_len and total 
    added importance is maximized.
    """

    n_segments = len(scores)
		
    k_table = np.zeros((n_segments + 1, final_len + 1))

    for seg_idx in range(1, n_segments + 1):
        for len_step in range(1, final_len + 1):
            if lens[seg_idx - 1] <= len_step:
                k_table[seg_idx, len_step] = max(
                    scores[seg_idx - 1] + 
                    k_table[seg_idx - 1, len_step - lens[seg_idx - 1]],
                    k_table[seg_idx - 1, len_step])
            else:
                k_table[seg_idx, len_step] = k_table[seg_idx - 1, len_step]

    segments = []
    len_left = final_len
    for seg_idx in range(n_segments, 0, -1):
        # print(f"seg {seg_idx} len {len_left}")
        if k_table[seg_idx, len_left] != k_table[seg_idx - 1, len_left]:
            segments.insert(0, seg_idx - 1)
            len_left -= lens[seg_idx - 1]

    return segments