import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    L = max_len if max_len is not None else (max(len(seq) for seq in seqs) if seqs else 0)
    for i in range(len(seqs)):
        if len(seqs[i]) > L: # Truncate
            seqs[i] = seqs[i][:L]
        elif len(seqs[i]) < L: # Pad
            seqs[i] += [pad_value] * (L - len(seqs[i]))
    return np.asarray(seqs)