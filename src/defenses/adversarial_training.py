"""
Adversarial Training Defense.

Strategy: At each epoch, generate adversarial examples from the
current model and append them to the training set. The model is then
re-trained on the augmented data, forcing it to learn features that
are robust against the given attack.

This is one of the most effective defenses against evasion attacks,
originally proposed by Goodfellow et al. (2015).
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.models.trainer import MODEL_REGISTRY, save_model
from configs.config import MODEL_PARAMS, RANDOM_STATE, DEFENSE_PARAMS


def adversarial_training(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    attack_type: str = "fgsm",
    epsilon: float = None,
    epochs: int = None,
) -> object:
    """
    Train a model with adversarial data augmentation.

    For each epoch:
      1. Use the current model to generate adversarial examples from the
         original training set.
      2. Concatenate the adversarial examples with the original training
         data (doubling the effective dataset size per epoch).
      3. Re-train the model from scratch on the augmented dataset.

    Args:
        model_name: One of 'random_forest', 'xgboost', 'lightgbm'.
        X_train, y_train: Original training data.
        X_val, y_val: Validation data (used for XGBoost eval_set).
        attack_type: 'fgsm' or 'pgd' — which attack to use for generation.
        epsilon: Perturbation magnitude for adversarial example generation.
        epochs: Number of augmentation / re-training cycles.

    Returns:
        The final adversarially-trained model.
    """
    # Read default parameters from config if not specified
    params = DEFENSE_PARAMS["adversarial_training"]
    epsilon = epsilon or params["epsilon"]
    epochs = epochs or params["epochs"]

    # Get the model class and base parameters
    model_cls = MODEL_REGISTRY[model_name]
    base_params = MODEL_PARAMS.get(model_name, {}).copy()
    base_params["random_state"] = RANDOM_STATE

    # Detect number of classes for XGBoost compatibility
    n_classes = len(np.unique(y_train))

    # Start with the original training pool
    X_adv_pool = X_train.copy()
    y_adv_pool = y_train.copy()

    # Train an initial model on clean data
    model = model_cls(**base_params)
    model.fit(X_train, y_train)

    # Augment-and-retrain loop
    for epoch in range(epochs):
        print(f"\n[adversarial_training] Epoch {epoch + 1}/{epochs}")

        # Generate adversarial examples from the *current* model
        if attack_type == "fgsm":
            X_adv = fgsm_attack(model, X_train, y_train, epsilon=epsilon)
        else:
            X_adv = pgd_attack(model, X_train, y_train, epsilon=epsilon)

        # Append adversarial examples to the training pool
        X_adv_pool = pd.concat([X_adv_pool, X_adv], axis=0).reset_index(drop=True)
        y_adv_pool = pd.concat([y_adv_pool, y_train], axis=0).reset_index(drop=True)

        # Re-train from scratch on the augmented dataset
        retrain_params = base_params.copy()
        if model_name == "xgboost":
            retrain_params["eval_metric"] = "logloss" if n_classes == 2 else "mlogloss"
            if n_classes > 2:
                retrain_params["num_class"] = n_classes
        model = model_cls(**retrain_params)
        eval_set = None
        if X_val is not None and y_val is not None and model_name == "xgboost":
            eval_set = [(X_val, y_val)]

        if eval_set:
            model.fit(X_adv_pool, y_adv_pool, eval_set=eval_set, verbose=False)
        else:
            model.fit(X_adv_pool, y_adv_pool)

        # Report progress
        train_acc = model.score(X_train, y_train)
        print(f"[adversarial_training] Epoch {epoch + 1} | Train acc: {train_acc:.4f}")

    # Save the robustified model
    save_model(model, model_name, "adv_trained")
    print(f"[adversarial_training] Robust model saved as '{model_name}_adv_trained.pkl'")
    return model
