# Project: eeg-adhd-honest-eval
Question: do published ADHD-vs-control EEG accuracies survive a child-level split?
- Data lives OUTSIDE the repo: C:\data\eeg\adhd\{ADHD,Control}\*.mat (19 ch, 128 Hz). Never commit data.
- Always split by CHILD, never by window. Windows from one child must never be in both train and test.
- Windows, conda env "medimg", CPU only.
-To learn: explain what you change and why, keep code simple and commented.