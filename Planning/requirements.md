# Requirements Document

## Introduction

This document specifies the requirements for a neural authorship attribution system that uses subword embedding techniques (BPE/WordPiece) combined with a CNN-LSTM architecture to classify social media documents to their authors. The system compares this deep learning approach against traditional baselines (BoW, TF-IDF, character n-grams, word n-grams) and integrates SHAP/LIME explainability to interpret model decisions and analyse misclassifications.

## Glossary

- **System**: The neural authorship attribution pipeline as a whole
- **DatasetLoader**: Component responsible for loading and splitting the social media dataset
- **Preprocessor**: Component that cleans and normalises raw social media text
- **SubwordTokeniser**: Component that tokenises text into subword units (BPE or WordPiece) and maps them to integer IDs
- **BaselineFeatureExtractor**: Component that extracts traditional features (BoW, TF-IDF, n-grams) for baseline comparison
- **CNNLSTMModel**: The primary deep learning model combining CNN and LSTM layers for authorship classification
- **Trainer**: Component managing the training loop, validation, early stopping, and checkpoint saving
- **Evaluator**: Component that computes classification metrics from model predictions
- **ExplainabilityModule**: Component that generates SHAP or LIME explanations for model predictions
- **BPE**: Byte Pair Encoding — a subword tokenisation algorithm
- **WordPiece**: An alternative subword tokenisation algorithm
- **Split**: A named partition of the dataset (train, val, or test)
- **MetricsDict**: Data structure holding accuracy, precision, recall, F1, per-class F1, and confusion matrix
- **ErrorAnalysisReport**: Output of the explainability module summarising misclassification patterns

---

## Requirements

### Requirement 1: Dataset Loading and Splitting

**User Story:** As a researcher, I want to load a social media authorship dataset and split it into train/validation/test partitions, so that I can train and evaluate models on consistent, non-overlapping data.

#### Acceptance Criteria

1. WHEN a valid CSV or JSON dataset file path is provided, THE DatasetLoader SHALL load all texts and map author names to integer label indices starting from 0.
2. WHEN `split()` is called with a dataset, THE DatasetLoader SHALL produce three non-overlapping partitions (train, val, test) with no sample appearing in more than one partition.
3. WHEN `split()` is called, THE DatasetLoader SHALL perform stratified splitting so that every author class present in the full dataset appears in the train, validation, and test splits.
4. WHEN an author class has fewer than the minimum sample threshold (default: 10) samples, THE DatasetLoader SHALL raise an `InsufficientSamplesError` identifying the offending author.
5. THE DatasetLoader SHALL expose dataset statistics including the number of authors and the number of samples per author.

---

### Requirement 2: Text Preprocessing

**User Story:** As a researcher, I want raw social media text to be cleaned and normalised before tokenisation, so that noise does not interfere with stylometric feature learning.

#### Acceptance Criteria

1. WHEN `clean()` is called on a raw text string, THE Preprocessor SHALL remove URLs, @mentions, and normalise whitespace.
2. WHEN `batch_clean()` is called on a list of texts, THE Preprocessor SHALL apply `clean()` to every text in the list and return a list of equal length.
3. WHEN a text becomes empty after cleaning, THE Preprocessor SHALL return an empty string and the System SHALL exclude that sample from further processing.
4. THE Preprocessor SHALL preserve punctuation patterns that carry stylometric signal unless explicitly configured otherwise.

---

### Requirement 3: Subword Tokenisation

**User Story:** As a researcher, I want text tokenised into subword units using BPE or WordPiece, so that the model can learn morphological and stylistic patterns at finer granularity than word-level features.

#### Acceptance Criteria

1. WHEN `train()` is called with a non-empty corpus and a target `vocab_size > 256`, THE SubwordTokeniser SHALL train a subword vocabulary using the specified algorithm (`"bpe"` or `"wordpiece"`).
2. WHEN BPE training completes, THE SubwordTokeniser SHALL produce a vocabulary whose size does not exceed `vocab_size`.
3. WHEN `encode()` is called on any string after training, THE SubwordTokeniser SHALL return a token ID sequence padded or truncated to exactly `max_length`.
4. WHEN `encode()` encounters a character not present in the trained vocabulary, THE SubwordTokeniser SHALL map it to the `[UNK]` token and continue encoding without raising an error.
5. WHEN a token sequence exceeds `max_length`, THE SubwordTokeniser SHALL truncate from the right to `max_length` tokens and log a warning with the original length.
6. WHEN `decode()` is called on a sequence of token IDs, THE SubwordTokeniser SHALL return a string that reconstructs the original text up to whitespace normalisation.
7. THE SubwordTokeniser SHALL persist the trained vocabulary to disk in a serialisable format for reproducibility.
8. THE SubwordTokeniser SHALL be trained exclusively on training corpus texts; validation and test texts SHALL only be encoded using `encode()` after training.

---

### Requirement 4: Baseline Feature Extraction

**User Story:** As a researcher, I want to extract traditional text features (BoW, TF-IDF, character n-grams, word n-grams), so that I can compare the CNN-LSTM model against established baselines.

#### Acceptance Criteria

1. WHEN `fit_transform()` is called with a list of texts and a method in `{"bow", "tfidf", "char_ngram", "word_ngram"}`, THE BaselineFeatureExtractor SHALL fit a vocabulary on those texts and return a sparse feature matrix of shape `[N, F]`.
2. WHEN `transform()` is called on unseen texts after fitting, THE BaselineFeatureExtractor SHALL apply the fitted vocabulary to return a sparse feature matrix without refitting.
3. THE BaselineFeatureExtractor SHALL support character n-gram ranges (e.g., 2–5) and word n-gram ranges (e.g., 1–3) as configurable parameters.
4. WHERE a fixed random seed is provided, THE BaselineFeatureExtractor SHALL produce identical feature matrices across runs given the same input texts and method.

---

### Requirement 5: CNN-LSTM Model

**User Story:** As a researcher, I want a CNN-LSTM model that learns author-specific stylometric representations from subword token sequences, so that I can classify documents to their authors with high accuracy.

#### Acceptance Criteria

1. WHEN `forward()` is called with a token ID tensor of shape `[B, T]` where `T >= max(kernel_sizes)`, THE CNNLSTMModel SHALL return a logits tensor of shape `[B, num_classes]`.
2. WHEN `forward()` is called, THE CNNLSTMModel SHALL apply parallel CNN filters over each configured kernel size with max-over-time pooling, producing one feature vector per kernel size per sample.
3. WHEN `forward()` is called, THE CNNLSTMModel SHALL concatenate the multi-scale CNN features and pass them through the stacked LSTM layers to produce a final hidden state.
4. WHEN `forward()` is called, THE CNNLSTMModel SHALL apply dropout at the embedding layer and before the classification head at the configured dropout rate.
5. IF any value in the output logits tensor is NaN or Inf, THEN THE CNNLSTMModel SHALL be considered to have failed a postcondition and the Trainer SHALL raise a `TrainingDivergenceError`.
6. THE CNNLSTMModel SHALL accept all hyperparameters (vocab_size, embed_dim, num_filters, kernel_sizes, lstm_hidden, lstm_layers, num_classes, dropout) at construction time via a `ModelConfig` object.

---

### Requirement 6: Model Training

**User Story:** As a researcher, I want a training loop with early stopping and checkpoint saving, so that I can train the CNN-LSTM model efficiently and recover the best-performing weights.

#### Acceptance Criteria

1. WHEN `train()` is called, THE Trainer SHALL optimise the model using the Adam optimiser with the specified learning rate and apply gradient clipping with `max_norm=1.0` before each parameter update.
2. WHEN `train()` is called, THE Trainer SHALL evaluate the model on the validation set after each epoch and compute macro-F1.
3. WHEN validation macro-F1 improves, THE Trainer SHALL save a model checkpoint to disk and reset the patience counter to 0.
4. WHEN the patience counter reaches the configured patience value without improvement, THE Trainer SHALL stop training before completing all epochs.
5. THE Trainer SHALL always terminate within the configured maximum number of epochs.
6. IF a NaN loss value is detected during training, THEN THE Trainer SHALL log the epoch and batch index and raise a `TrainingDivergenceError`.
7. IF a CUDA out-of-memory error occurs, THEN THE Trainer SHALL halve the batch size and retry the current epoch, logging a warning.
8. WHEN training completes, THE Trainer SHALL return a `TrainingHistory` object containing per-epoch loss and validation metrics.

---

### Requirement 7: Model Evaluation

**User Story:** As a researcher, I want to evaluate trained models using standard classification metrics, so that I can compare the CNN-LSTM model against baselines and report results.

#### Acceptance Criteria

1. WHEN `evaluate()` is called with a trained model and a labelled dataset, THE Evaluator SHALL compute accuracy, macro-precision, macro-recall, macro-F1, per-class F1, and a confusion matrix.
2. THE Evaluator SHALL return all scalar metric values (accuracy, precision, recall, F1) in the range `[0.0, 1.0]`.
3. WHEN a confusion matrix is computed, THE Evaluator SHALL produce a matrix where the sum of all elements equals the total number of evaluated samples.
4. WHEN a confusion matrix is computed, THE Evaluator SHALL ensure that the trace of the confusion matrix divided by the total number of samples equals the accuracy.
5. WHEN `evaluate()` is called, THE Evaluator SHALL run inference in no-gradient mode to avoid modifying model weights.

---

### Requirement 8: Explainability and Error Analysis

**User Story:** As a researcher, I want SHAP and LIME explanations for model predictions on misclassified samples, so that I can understand systematic failure patterns and interpret model decisions.

#### Acceptance Criteria

1. WHEN `explain_shap()` is called with a model, tokeniser, texts, and background texts, THE ExplainabilityModule SHALL compute token-level SHAP attributions for each provided text.
2. WHEN `explain_lime()` is called with a model, tokeniser, and a single text, THE ExplainabilityModule SHALL compute a local surrogate LIME explanation for that text.
3. WHEN `error_analysis()` is called with explanations, predictions, and labels, THE ExplainabilityModule SHALL produce an `ErrorAnalysisReport` containing attributions for every misclassified sample in the input.
4. WHEN `error_analysis()` is called, THE ExplainabilityModule SHALL identify the top-k most influential subword tokens per misclassified author pair.
5. WHEN `error_analysis()` is called, THE ExplainabilityModule SHALL identify the author pairs with the highest confusion rate.
6. IF `error_analysis()` is called with no misclassified samples, THEN THE ExplainabilityModule SHALL raise an error indicating that at least one misclassified sample is required.

---

### Requirement 9: End-to-End Pipeline

**User Story:** As a researcher, I want a complete, reproducible pipeline from raw data to evaluation results, so that I can run experiments consistently and compare conditions.

#### Acceptance Criteria

1. WHEN the full pipeline is executed with a fixed random seed, THE System SHALL produce identical metric results across runs on the same hardware.
2. WHEN the pipeline completes, THE System SHALL save evaluation metrics to `results/metrics.json` for each experimental condition.
3. WHEN the pipeline completes, THE System SHALL save the trained tokeniser vocabulary to `artifacts/tokeniser.json`.
4. WHEN the pipeline completes, THE System SHALL save model checkpoints to `artifacts/checkpoints/`.
5. WHEN the pipeline is run on a synthetic dataset of at least 50 samples across 5 authors, THE System SHALL complete without error and produce metric values in `[0.0, 1.0]`.
6. THE System SHALL produce both CNN-LSTM and baseline evaluation results in a single pipeline run for direct comparison.

---

### Requirement 10: Subword Tokeniser Parsing and Serialisation

**User Story:** As a researcher, I want the subword tokeniser to correctly encode and decode text in a round-trip manner, so that I can verify tokenisation correctness and use decoded tokens in explainability outputs.

#### Acceptance Criteria

1. WHEN a valid text string is encoded and then decoded, THE SubwordTokeniser SHALL reconstruct a string equivalent to the original up to whitespace normalisation.
2. THE SubwordTokeniser SHALL serialise its trained vocabulary and merge rules to disk in a format that can be fully deserialised to restore identical encoding behaviour.
3. WHEN a serialised tokeniser is deserialised and used to encode the same text, THE SubwordTokeniser SHALL produce the same token ID sequence as the original tokeniser instance.
