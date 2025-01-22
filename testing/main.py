import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import nltk
from nltk.corpus import wordnet as wn
import yaml
import argparse

def download_nltk_dependencies():
    """
    Downloads necessary NLTK data files.
    """
    nltk_packages = ['wordnet', 'omw-1.4']
    for package in nltk_packages:
        try:
            nltk.download(package, quiet=True)
            print(f"Downloaded {package}")
        except Exception as e:
            print(f"Failed to download {package}")
            print(f"Error: {e}")

def get_words(pos, max_words=1000, max_length=5):
    """
    Retrieve a list of simple words for a specific part of speech from WordNet.
    """
    words = set()
    try:
        for synset in wn.all_synsets(pos=pos):
            for lemma in synset.lemmas():
                word = lemma.name().replace('_', ' ').lower()
                if word.isalpha() and 2 < len(word) <= max_length:
                    words.add(word)
                if len(words) >= max_words:
                    break
            if len(words) >= max_words:
                break
    except Exception as e:
        print(f"Error retrieving {pos} words: {e}")
    return list(words)

def generate_model_name(prefix='v6', adjectives=[], nouns=[], random_state=None):
    """
    Generates a unique model name in the format 'v6-adj-noun-rs{random_state}'.
    """
    try:
        adjective = random.choice(adjectives)
        noun = random.choice(nouns)
        if random_state is not None:
            model_name = f"{prefix}-{adjective}-{noun}-rs{random_state}"
        else:
            model_name = f"{prefix}-{adjective}-{noun}"
        return model_name
    except IndexError:
        print("Error: Adjective or noun list is empty. Returning default name.")
        return f"{prefix}-model"

def create_plots_dir(base_dir):
    """
    Creates a directory for saving plots if it doesn't exist.
    """
    plots_dir = os.path.join(base_dir, 'plots')
    try:
        os.makedirs(plots_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating plots directory: {e}")
    return plots_dir

def save_plot(fig, filepath):
    """
    Saves a matplotlib figure to the specified filepath.
    """
    try:
        fig.savefig(filepath)
        plt.close(fig)
    except Exception as e:
        print(f"Error saving plot {filepath}: {e}")

def main(random_state, config):
    """
    Main training routine for the model.
    """
    # Setup
    league_name = config['league']
    data_path = config['training_dataset_pathway']['Match_Result']
    metrics_dir_base = config['training_model_pathways']['Match_Result']

    print("Starting training...")

    # Download NLTK dependencies
    download_nltk_dependencies()

    # Get lists of adjectives and nouns
    adjectives = get_words('a', max_words=1000, max_length=7)
    nouns = get_words('n', max_words=1000, max_length=5)

    if not adjectives or not nouns:
        print("Error: No adjectives or nouns found. Exiting.")
        return

    # Generate a unique model name
    model_base_name = generate_model_name(prefix='v6', adjectives=adjectives, nouns=nouns, random_state=random_state)

    # Define paths
    metrics_dir = os.path.join(metrics_dir_base, model_base_name)
    model_filename = os.path.join(metrics_dir, f"{model_base_name}.joblib")
    metrics_filename = os.path.join(metrics_dir, "metrics.json")
    plots_dir = create_plots_dir(metrics_dir)

    # Create metrics directory
    try:
        os.makedirs(metrics_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating metrics directory: {e}")
        return

    # Model Notes
    model_notes = f"""
    Model Notes:
    ------------
    - Uses two rows per match to include 'is_home'.
    - Random state: {random_state}
    - League: {league_name}
    """

    # Save Notes
    notes_filename = os.path.join(metrics_dir, "notes.txt")
    try:
        with open(notes_filename, 'w') as notes_file:
            notes_file.write(model_notes)
    except Exception as e:
        print(f"Error saving notes: {e}")
        return

    # Load and Prepare Data
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded dataset from {data_path}")
    except Exception as e:
        print(f"Error loading dataset from {data_path}: {e}")
        return

    # Fill missing 'winning_team' with 0
    df["winning_team"] = df["winning_team"].fillna(0)

    # Separate classes
    draws = df[df["winning_team"] == 0]
    home_wins = df[df["winning_team"] == 2]
    away_wins = df[df["winning_team"] == 1]

    # Determine min samples for balancing
    min_samples = min(len(draws), len(home_wins), len(away_wins))

    # Resample
    try:
        home_wins_downsampled = home_wins.sample(n=min_samples, random_state=random_state)
        draws_upsampled = draws.sample(n=min_samples, replace=True, random_state=random_state)
        away_wins_downsampled = away_wins.sample(n=min_samples, replace=True, random_state=random_state)
    except Exception as e:
        print(f"Error during resampling: {e}")
        return

    # Combine and shuffle
    df_balanced = pd.concat([home_wins_downsampled, away_wins_downsampled, draws_upsampled])
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Define features
    features = [
        'team_id', 'opponent_id',
        'team_ELO_before', 'opponent_ELO_before',
        'odds_team_win',
        'odds_draw',
        'odds_opponent_win',
        'opponent_rest_days', 'team_rest_days',
        'team_h2h_win_percent', 'opponent_h2h_win_percent',
        'pre_match_home_ppg', 'pre_match_away_ppg',
        'team_home_advantage', 'opponent_home_advantage'
    ]

    X = df_balanced[features]
    y = df_balanced["winning_team"]

    # Store match details
    match_details = df_balanced[["team_id", "opponent_id", "match_id"]]

    # Initialize Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=165,
        random_state=random_state,
        min_samples_leaf=1,
        n_jobs=-1
    )

    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    cv_accuracies = []
    cv_precisions = []
    cv_recalls = []
    cv_f1s = []
    feature_importances = np.zeros(X.shape[1])

    # For storing predictions
    predictions = []
    confidences = []
    actuals = []
    home_ids = []
    away_ids = []
    match_ids = []

    # Class labels for clarity
    class_labels = [0, 1, 2]  # 0: Draw, 1: Away Win, 2: Home Win
    class_names = ['Draw', 'Away Win', 'Home Win']

    fold_num = 1
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average='macro')
        recall = recall_score(y_test, preds, average='macro')
        f1 = f1_score(y_test, preds, average='macro')

        cv_accuracies.append(acc)
        cv_precisions.append(precision)
        cv_recalls.append(recall)
        cv_f1s.append(f1)

        # Store predictions/confidences/actuals
        predictions.extend(preds)
        actuals.extend(y_test)
        confidences.extend(np.max(probas, axis=1))

        # Store match details
        home_ids.extend(match_details.iloc[test_index]["team_id"])
        away_ids.extend(match_details.iloc[test_index]["opponent_id"])
        match_ids.extend(match_details.iloc[test_index]["match_id"])

        feature_importances += model.feature_importances_

        # Optional: print fold-level info
        print(f"Fold {fold_num}: Accuracy={acc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        fold_num += 1

    # Average feature importances
    feature_importances /= skf.get_n_splits()

    # Compile Results
    results_df = pd.DataFrame({
        'Actual': actuals,
        'Predicted': predictions,
        'Confidence': confidences,
        'HomeID': home_ids,
        'AwayID': away_ids,
        'MatchID': match_ids
    })

    # Overall Accuracy
    overall_accuracy = (results_df['Actual'] == results_df['Predicted']).mean()
    print(f"Overall Accuracy: {overall_accuracy:.3f}")

    # Classification Report
    report = classification_report(results_df['Actual'], results_df['Predicted'], output_dict=True, zero_division=0)
    overall_precision = report['macro avg']['precision']
    overall_recall = report['macro avg']['recall']
    overall_f1 = report['macro avg']['f1-score']

    # Confusion Matrix
    cm = confusion_matrix(results_df['Actual'], results_df['Predicted'], labels=class_labels)

    # High Confidence filtering
    high_confidence_threshold = 0.51
    high_confidence_df = results_df[results_df['Confidence'] >= high_confidence_threshold]

    if len(high_confidence_df) > 0:
        high_conf_report = classification_report(
            high_confidence_df['Actual'],
            high_confidence_df['Predicted'],
            output_dict=True,
            zero_division=0
        )
        high_confidence_accuracy = (high_confidence_df['Actual'] == high_confidence_df['Predicted']).mean()
        high_conf_precision = high_conf_report['macro avg']['precision']
        high_conf_recall = high_conf_report['macro avg']['recall']
        high_conf_f1 = high_conf_report['macro avg']['f1-score']
        print(f"High-Confidence Accuracy (>= {high_confidence_threshold*100}%): {high_confidence_accuracy:.3f}")
    else:
        high_confidence_accuracy = 0
        high_conf_precision = 0
        high_conf_recall = 0
        high_conf_f1 = 0
        print("No high-confidence predictions found.")

    # Print cross-validation statistics
    mean_cv_accuracy = np.mean(cv_accuracies)
    std_cv_accuracy = np.std(cv_accuracies)
    mean_cv_precision = np.mean(cv_precisions)
    std_cv_precision = np.std(cv_precisions)
    mean_cv_recall = np.mean(cv_recalls)
    std_cv_recall = np.std(cv_recalls)
    mean_cv_f1 = np.mean(cv_f1s)
    std_cv_f1 = np.std(cv_f1s)

    print(f"CV Accuracy: {mean_cv_accuracy:.3f} ± {std_cv_accuracy:.3f}")
    print(f"CV Precision: {mean_cv_precision:.3f} ± {std_cv_precision:.3f}")
    print(f"CV Recall: {mean_cv_recall:.3f} ± {std_cv_recall:.3f}")
    print(f"CV F1: {mean_cv_f1:.3f} ± {std_cv_f1:.3f}")

    # Retrain on entire dataset
    final_model = RandomForestClassifier(n_estimators=165, random_state=random_state, min_samples_leaf=1)
    final_model.fit(X, y)
    print("Retrained final model on entire dataset.")

    # Save model
    try:
        joblib.dump(final_model, model_filename)
        print(f"Model saved to {model_filename}")
    except Exception as e:
        print(f"Error saving model: {e}")

    # Save feature importances
    feature_importances_dict = dict(zip(features, feature_importances))
    try:
        with open(os.path.join(metrics_dir, "feature_importances.json"), 'w') as f:
            json.dump(feature_importances_dict, f, indent=4)
    except Exception as e:
        print(f"Error saving feature importances: {e}")

    # Plot and save confusion matrix
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        cm_plot_path = os.path.join(plots_dir, 'confusion_matrix.png')
        save_plot(plt.gcf(), cm_plot_path)
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

    # Plot and save feature importances
    try:
        plt.figure(figsize=(10, 8))
        sorted_features = sorted(feature_importances_dict.items(), key=lambda x: x[1], reverse=True)
        features_sorted, importances_sorted = zip(*sorted_features)
        sns.barplot(x=importances_sorted, y=features_sorted, palette='viridis')
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importances')
        fi_plot_path = os.path.join(plots_dir, 'feature_importances.png')
        save_plot(plt.gcf(), fi_plot_path)
    except Exception as e:
 
