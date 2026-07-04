# Embedding Training Specification

## Overview
Fine-tunes sentence-transformer models on Alexa QA pairs using contrastive loss. Improves distance accuracy for SOT routing decisions.

**Implementation:** `src/train/embedder.py`

---

## Requirements

### Requirement: Training Data Preparation
The system SHALL prepare contrastive training pairs from the Alexa QA dataset.

#### Scenario: Prepare pairs
- **WHEN** `prepare_training_data(max_pairs)` is called
- **THEN** load QA pairs from downloaded CSV
- **AND** create contrastive triplets: (anchor, positive, negative)
  - anchor: question text
  - positive: matching answer (should be close)
  - negative: random different answer (should be far)
- **AND** return up to `max_pairs` triplets

#### Scenario: No real data available
- **WHEN** no QA data is available
- **THEN** generate synthetic training pairs
- **AND** log that synthetic data is being used

---

### Requirement: Model Fine-Tuning
The system SHALL fine-tune a sentence-transformer model on the prepared data.

#### Scenario: Training runs
- **WHEN** `train(pairs)` is called
- **THEN** load base model (all-MiniLM-L6-v2)
- **AND** create contrastive loss training objective
- **AND** train for N epochs with configurable batch size and learning rate
- **AND** save fine-tuned model to `data/models/{output_name}/`

#### Scenario: No sentence-transformers
- **WHEN** sentence-transformers is not installed
- **THEN** log error explaining what's needed
- **AND** do not crash

---

### Requirement: Distance Evaluation
The system SHALL evaluate the trained model's distance accuracy.

#### Scenario: Evaluate model
- **WHEN** `evaluate_distance_accuracy(model_path)` is called
- **THEN** compute cosine distances for known similar/dissimilar pairs
- **AND** return accuracy, TP, TN, FP, FN
- **AND** log per-pair results

---

### Requirement: Full Training Pipeline
The system SHALL run end-to-end training in one command.

#### Scenario: Run training pipeline
- **WHEN** `run_training_pipeline(epochs, max_pairs)` is called
- **THEN** execute: prepare data → train model → evaluate accuracy
- **AND** return path to saved model

---

## Success Criteria

- [ ] Contrastive triplets created from QA pairs
- [ ] Model training runs without errors
- [ ] Fine-tuned model saved to disk
- [ ] Evaluation metrics returned
- [ ] Synthetic data fallback works
- [ ] Full pipeline runs end-to-end
