MIMIC-III / MIMIC-IV access requirements
---------------------------------------
Real MIMIC data is credentialed on PhysioNet and MUST NOT be auto-downloaded
by this prototype.

1. Complete the CITI "Data or Specimens Only Research" training.
2. Sign the PhysioNet credentialed health data use agreement.
3. Request access to MIMIC-III or MIMIC-IV on https://physionet.org/
4. Place approved extracts under code/data/mimic/ (never commit PHI).
5. Extend datasets/mimic.py::load_mimic() to read your local extract.

Until then, `--dataset mimic` uses a tiny synthetic MIMIC-shaped sample only.
