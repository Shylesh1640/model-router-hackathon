"""Embedding model trainer — fine-tunes on Alexa QA data for better SOT distance."""

import json
import logging
import math
import random
from pathlib import Path
from typing import Optional

from src.data.scraper import download_alexa_qa, DATA_DIR

logger = logging.getLogger(__name__)

MODEL_DIR = DATA_DIR / "models"


# =============================================================================
# CONTRASTIVE TRAINING DATA
# =============================================================================

def prepare_training_data(max_pairs: int = 5000) -> list[dict]:
    """Prepare contrastive training pairs from Alexa QA dataset.

    Each pair: {anchor, positive, negative}
    - anchor: question
    - positive: matching answer
    - negative: random different answer
    """
    path = download_alexa_qa()
    if not path:
        logger.warning("No Alexa QA data, generating synthetic training data")
        return _synthetic_data()

    pairs = []

    try:
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        questions = [table.column("question")[i].as_py() for i in range(len(table))]
        answers = [table.column("answer")[i].as_py() for i in range(len(table))]
    except Exception:
        json_path = path.with_suffix(".json")
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            questions = [d.get("question", "") for d in data]
            answers = [d.get("answer", "") for d in data]
        else:
            logger.warning("Cannot load QA data, using synthetic")
            return _synthetic_data()

    if not questions:
        return _synthetic_data()

    for i in range(min(len(questions), max_pairs)):
        if not questions[i] or not answers[i]:
            continue
        # Pick a random different question as negative
        j = random.choice([k for k in range(len(answers)) if k != i])
        pairs.append({
            "anchor": questions[i],
            "positive": answers[i],
            "negative": answers[j] if j < len(answers) else answers[0],
        })

    logger.info(f"Prepared {len(pairs)} training pairs")
    return pairs


def _synthetic_data() -> list[dict]:
    """Generate synthetic training pairs for dev/testing."""
    topics = [
        ("python", "programming language", "cooking recipe"),
        ("docker", "container platform", "music genre"),
        ("api", "interface for services", "sports team"),
        ("database", "structured data storage", "movie title"),
        ("machine learning", "AI subset", "automobile brand"),
        ("linux", "open source OS", "dessert name"),
        ("javascript", "web programming language", "furniture type"),
        ("git", "version control", "plant species"),
        ("cloud", "remote computing", "clothing brand"),
        ("algorithm", "step-by-step procedure", "animal habitat"),
    ]
    pairs = []
    for anchor, positive, negative in topics:
        pairs.append({"anchor": anchor, "positive": positive, "negative": negative})
    logger.info(f"Generated {len(pairs)} synthetic training pairs")
    return pairs


# =============================================================================
# EMBEDDING MODEL TRAINING
# =============================================================================

class EmbeddingTrainer:
    """Fine-tunes a sentence-transformer model on QA pairs.

    Uses contrastive loss to push similar Q-A pairs together
    and different Q-A pairs apart in embedding space.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._loaded = False

    def load_model(self):
        """Load the base embedding model."""
        if self._loaded:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded model: {self.model_name} ({self.model.get_sentence_embedding_dimension()}d)")
            self._loaded = True
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def train(
        self,
        pairs: list[dict],
        output_name: str = "sot-embedder",
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ) -> Path:
        """Fine-tune the embedding model on contrastive pairs.

        Args:
            pairs: [{anchor, positive, negative}, ...]
            output_name: Name for the saved model
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Adam learning rate

        Returns:
            Path to saved model
        """
        self.load_model()

        from sentence_transformers import InputExample, losses
        from torch.utils.data import DataLoader

        # Prepare training examples
        train_examples = []
        for p in pairs:
            if not p.get("anchor") or not p.get("positive"):
                continue
            # Anchor-positive as similar (label=1)
            train_examples.append(InputExample(
                texts=[p["anchor"], p["positive"]],
                label=1.0,
            ))
            # Anchor-negative as dissimilar (label=0)
            if p.get("negative"):
                train_examples.append(InputExample(
                    texts=[p["anchor"], p["negative"]],
                    label=0.0,
                ))

        if not train_examples:
            logger.warning("No training examples to train on")
            return self._save_model(output_name)

        logger.info(f"Training on {len(train_examples)} examples, {epochs} epochs")

        # DataLoader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size,
        )

        # Contrastive loss
        train_loss = losses.ContrastiveLoss(self.model)

        # Train
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=int(len(train_dataloader) * epochs * 0.1),
            optimizer_params={"lr": learning_rate},
            show_progress_bar=True,
        )

        # Save
        output_path = self._save_model(output_name)
        logger.info(f"Training complete. Model saved to {output_path}")
        return output_path

    def _save_model(self, name: str) -> Path:
        """Save the fine-tuned model."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        output_path = MODEL_DIR / name
        self.model.save(str(output_path))
        return output_path


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_distance_accuracy(model_path: Optional[Path] = None) -> dict:
    """Evaluate how well the embedding model separates related vs unrelated pairs.

    Returns metrics dict.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        if model_path and model_path.exists():
            model = SentenceTransformer(str(model_path))
            logger.info(f"Loaded fine-tuned model from {model_path}")
        else:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded base model for evaluation")

    except ImportError:
        logger.error("sentence-transformers not available")
        return {"error": "sentence-transformers not installed"}

    # Test pairs
    test_pairs = [
        ("what is python", "python is a programming language", True),
        ("how to sort list", "use list.sort() method", True),
        ("what is docker", "docker is a container platform", True),
        ("what is python", "i like pizza", False),
        ("how to sort list", "the weather is nice", False),
        ("what is docker", "my favorite color is blue", False),
    ]

    tp, tn, fp, fn = 0, 0, 0, 0
    for a, b, expected_similar in test_pairs:
        emb_a = model.encode(a)
        emb_b = model.encode(b)
        sim = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))
        distance = 1.0 - sim

        is_close = distance < 0.5
        if is_close == expected_similar:
            if expected_similar:
                tp += 1
            else:
                tn += 1
        else:
            if not expected_similar:
                fp += 1
            else:
                fn += 1

        tag = "✓" if is_close == expected_similar else "✗"
        logger.info(f"  {tag} d={distance:.3f} (expect similar={expected_similar}): {a[:30]} → {b[:30]}")

    accuracy = (tp + tn) / len(test_pairs) if test_pairs else 0
    results = {
        "accuracy": accuracy,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": len(test_pairs),
    }
    logger.info(f"Distance accuracy: {accuracy:.1%}")
    return results


# =============================================================================
# CLI
# =============================================================================

def run_training_pipeline(epochs: int = 3, max_pairs: int = 5000):
    """Full training pipeline: prepare data → train → evaluate."""
    logger.info("=== Embedding Training Pipeline ===")

    # Step 1: Prepare training data
    logger.info("Step 1: Preparing training data...")
    pairs = prepare_training_data(max_pairs=max_pairs)
    logger.info(f"  {len(pairs)} pairs ready")

    # Step 2: Train
    logger.info("Step 2: Training embedding model...")
    trainer = EmbeddingTrainer()
    model_path = trainer.train(pairs, epochs=epochs)
    logger.info(f"  Model saved to {model_path}")

    # Step 3: Evaluate
    logger.info("Step 3: Evaluating...")
    metrics = evaluate_distance_accuracy(model_path)
    logger.info(f"  Accuracy: {metrics.get('accuracy', 'N/A')}")

    logger.info("=== Training Complete ===")
    return model_path
