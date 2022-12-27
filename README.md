# Speaker Recognition

This is a small speaker identification application developed as course work at SPbSTU.

The goal is to extract audio features from WAV recordings, train a **multilayer perceptron** to recognize speakers, and evaluate its performance.

## Modules
- **Feature Extraction**:  
  - *FeatureExtractor.py* uses LibROSA. Extracts:
    - chroma
    - RMS
    - spectral centroid
    - bandwidth
    - rolloff
    - zero-crossing rate
    - 20 MFCCs

- **Data Preprocessing**:  
  - *PreprocessData.py* loads the CSV, assigns a numeric label to each speaker, returns a cleaned DataFrame.

- **Speaker Recognition**:  
  - *speaker-recognition.ipynb* documents the workflow in Jupyter: feature extraction, data split, normalization, model training and evaluation.
