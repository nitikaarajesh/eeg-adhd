"""Find and load the ADHD/Control recordings."""
from pathlib import Path
from scipy.io import loadmat

ROOT = Path(r"C:\data\eeg\adhd")

def list_children():
    """One row per child: id, group, label (1 = ADHD, 0 = Control), file path."""
    rows = []
    for group, label in [("ADHD", 1), ("Control", 0)]:
        for f in sorted((ROOT / group).glob("*.mat")):
            rows.append({"child_id": f.stem, "group": group, "label": label, "path": str(f)})
    return rows

def load_child(path):
    """Return the EEG as a channels x samples array in µV."""
    m = loadmat(path)
    key = [k for k in m if not k.startswith("__")][0]
    return m[key].T.astype(float)