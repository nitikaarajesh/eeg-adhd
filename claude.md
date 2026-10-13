# Project: eeg-adhd-honest-eval
Question: do published ADHD-vs-control EEG accuracies survive a subject-wise split?
- Data lives OUTSIDE the repo: C:\data\eeg\adhd\{ADHD,Control}\*.mat (19 ch, 128 Hz). Never commit data.
- Always split by SUBJECT, never by window. Windows from one subject must never be in both train and test.
- Windows, conda env "medimg", CPU only.
-To learn: explain what you change and why, keep code simple and commented.