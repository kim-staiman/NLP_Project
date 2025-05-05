import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # Import AdamW directly from torch.optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    RobertaModel,
    XLMRobertaModel,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    get_cosine_schedule_with_warmup,
    BertModel,
    BertTokenizer,
    BertForSequenceClassification
)
from torch import nn
from tqdm import tqdm

os.environ["USE_TF"] = "0"

# Define constants
MAX_LEN = 512
BATCH_SIZE = 64
EPOCHS = 30
HIGH_LEARNING_RATE = 1e-3
LOW_LEARNING_RATE = 2e-5
# LANGUAGE_MAP = {
#   'es': 0,  # Spanish
#   'fr': 1,  # French
#   'it': 2,  # Italian
#   'de': 3  # German
# }
# LANGUAGE_MAP = {
#    'el': 0,  # Greek
#    'fr': 1,  # French
#    'fi': 2,  # Finnish
#    'de': 3  # German
# }
LANGUAGE_MAP = {
    'fr': 0,  # French
    'ja': 1  # Japanese
}
NUM_LABELS = len(LANGUAGE_MAP)
# TODO: change for XLM/BERT
MODEL_NAME = "roberta-base"
#MODEL_NAME = "xlm-roberta-base"
#MODEL_NAME = "bert-base-uncased"


class RobertaWithNonLinearClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
        super().__init__()
        # TODO: change for XLM/BERT
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
        #self.roberta = XLMRobertaModel.from_pretrained(pretrained_model_name)
        #self.roberta = BertModel.from_pretrained(pretrained_model_name)

        # Freeze all parameters of the base RoBERTa model
        for param in self.roberta.parameters():
            param.requires_grad = False

        # Non-linear classification head with proper initialization
        hidden_size = self.roberta.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )

        # Initialize the classifier weights properly
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Initialize with larger variance to create more separation
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.classifier.apply(init_weights)

        # Print info about freezing
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Percentage trainable: {trainable_params / total_params * 100:.2f}%")

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Mean pooling instead of CLS token
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()  # Expanded from shape (batch_size, sequence_len) to (batch_size, sequence_len, hidden_size)
        pooled_output = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)  # Mean of non-padding tokens

        # CLS token
        # pooled_output = outputs.last_hidden_state[:, 0, :]  # Take [CLS] token representation
        # pooled_output = self.dropout(pooled_output)  # Dropout for regularization

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(logits.view(-1, self.classifier[-1].out_features), labels.view(-1))

        return type('ModelOutput', (), {'loss': loss, 'logits': logits})()


class TranslationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Function to load and prepare data
def prepare_data(data_file, languages, lang_map=LANGUAGE_MAP):
    """
    Load and prepare data from TSV file with columns: translated, origin_language
    """
    df = pd.read_csv(data_file, sep='\t')

    # Filter for the languages we want to include
    df = df[df['origin_language'].isin(languages)]

    # Map language codes to numerical labels
    df['language'] = df['origin_language'].map(lang_map)

    df = df.rename(columns={'translated': 'text'})

    # Shuffle the data
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def train_model(model, train_dataloader, val_dataloader, device, epochs, initial_lr=HIGH_LEARNING_RATE, min_lr=LOW_LEARNING_RATE):
    """
    Train the model and validate on the validation set
    """
    weight_decay = 0.05
    unfreeze_after_epoch = 15
    patience = 7  # Optional - used for early stopping

    # Set up optimizer with different learning rates for classifier and base model
    optimizer_grouped_params = [
        {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 'lr': min_lr},
        {'params': [p for n, p in model.named_parameters() if 'classifier' in n], 'lr': initial_lr}
    ]
    optimizer = AdamW(optimizer_grouped_params, weight_decay=weight_decay)

    best_val_loss = float('inf')
    best_model_state = None
    no_improvement = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        if epoch == unfreeze_after_epoch:  # Unfreeze roBERTa parameters
            print("Before unfreezing, loading best model...")
            # Load the best model state if we have one
            if best_model_state is not None:
                print(
                    f"Loading best model from epoch {best_model_state['epoch']} "
                    f"with validation loss: {best_model_state['val_loss']:.4f} "
                    f"with validation accuracy: {best_model_state['val_accuracy']:.4f}")
                model.load_state_dict(best_model_state['model_state_dict'])

            # Unfreeze all parameters of the base RoBERTa model
            for param in model.roberta.parameters():
                param.requires_grad = True
            # Print updated parameter counts
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"After unfreezing, trainable parameters: {trainable_params}")
            print(f"Percentage trainable: {trainable_params / total_params * 100:.2f}%")

            optimizer = AdamW(model.parameters(), lr=min_lr, weight_decay=weight_decay)
            total_steps = len(train_dataloader) * (epochs - unfreeze_after_epoch)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.2 * total_steps),  # 20% warmup
                num_training_steps=total_steps
            )

        model.train()
        total_loss = 0
        train_predictions = []
        train_labels_list = []

        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            train_predictions.extend(preds.cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())

            progress_bar.set_postfix({'loss': loss.item()})

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            if epoch > unfreeze_after_epoch:
                scheduler.step()

        # Calculate average training loss and metrics
        avg_train_loss = total_loss / len(train_dataloader)
        train_accuracy = accuracy_score(train_labels_list, train_predictions)
        train_f1 = f1_score(train_labels_list, train_predictions, average='macro')

        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Training Macro F1: {train_f1:.4f}")

        # Evaluate on validation set
        print("\nValidation Set Metrics:")
        val_metrics = evaluate_model(model, val_dataloader, device)
        print(f"Validation Loss: {val_metrics['loss']}")
        print(f"Validation Accuracy: {val_metrics['accuracy']}")
        print(f"Validation Macro F1: {val_metrics['macro_f1']}")
        print(f"Validation Macro Precision: {val_metrics['macro_precision']}")
        print(f"Validation Macro Recall: {val_metrics['macro_recall']}")
        print(f"Validation Per-class F1: {val_metrics['per_class_f1']}")
        print(f"Validation Per-class Precision: {val_metrics['per_class_precision']}")
        print(f"Validation Per-class Recall: {val_metrics['per_class_recall']}")

        print("\nValidation Confusion Matrix:")
        cm_val = val_metrics['confusion_matrix']
        present_classes_val = val_metrics['present_classes']
        inv_map = {v: k for k, v in LANGUAGE_MAP.items()}
        # Use only the classes present in the data for the confusion matrix
        lang_labels_val = [inv_map.get(idx, f"Unknown_{idx}") for idx in present_classes_val]
        cm_df_val = pd.DataFrame(cm_val, index=lang_labels_val, columns=lang_labels_val)
        print(cm_df_val)

        # Check if this is the best model so far
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            # Save model state
            best_model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': val_metrics['accuracy'],
                'val_f1': val_metrics['macro_f1']
            }
            no_improvement = 0
            print(f"✓ New best model saved!")
        else:
            no_improvement += 1
            print(f"No improvement for {no_improvement} epochs.")

        # Early stopping
        # if no_improvement >= patience:
        #    print(f"Early stopping after {epoch + 1} epochs with no improvement.")
        #    break

    # Load the best model state if we have one
    if best_model_state is not None:
        print(
            f"Loading best model from epoch {best_model_state['epoch']} "
            f"with validation loss: {best_model_state['val_loss']:.4f} "
            f"with validation accuracy: {best_model_state['val_accuracy']:.4f}")
        model.load_state_dict(best_model_state['model_state_dict'])

    return model


def evaluate_model(model, dataloader, device, num_labels=NUM_LABELS, lang_map=LANGUAGE_MAP):
    """
    Evaluate the model on the provided dataloader
    """
    # Set the model to evaluation mode
    model.eval()

    val_loss = 0
    predictions = []
    true_labels = []

    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            val_loss += outputs.loss.item()

            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            # Add predictions and true labels for metric calculation
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            total_examples += len(batch['labels'])

    all_classes = list(range(num_labels))  # All classes in data

    # Find which classes are present in the true labels or predictions
    present_classes = sorted(set(true_labels) | set(predictions))

    # Handle case when some classes might be missing from both true labels and predictions
    if len(present_classes) < num_labels:
        print(f"Warning: Only {len(present_classes)} out of {num_labels} classes present in this evaluation")
        print(f"Present classes: {[int(i) for i in present_classes]}")
        print(f"Classes in batch: {sorted(set([int(i) for i in true_labels]))}, predicted classes: {sorted(set([int(i) for i in predictions]))}")

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, labels=present_classes, average='macro', zero_division=0)
    per_class_f1 = f1_score(true_labels, predictions, labels=present_classes, average=None, zero_division=0)

    macro_precision = precision_score(true_labels, predictions, labels=present_classes, average='macro', zero_division=0)
    macro_recall = recall_score(true_labels, predictions, labels=present_classes, average='macro', zero_division=0)

    per_class_precision = precision_score(true_labels, predictions, labels=present_classes, average=None, zero_division=0)
    per_class_recall = recall_score(true_labels, predictions, labels=present_classes, average=None, zero_division=0)

    # Map indices back to language codes for reporting
    inv_map = {v: k for k, v in lang_map.items()}

    per_class_precision_dict = {}
    per_class_recall_dict = {}
    per_class_f1_dict = {}

    for i, class_idx in enumerate(present_classes):
        lang_code = inv_map.get(class_idx, f"Unknown_{class_idx}")
        per_class_precision_dict[lang_code] = per_class_precision[i]
        per_class_recall_dict[lang_code] = per_class_recall[i]
        per_class_f1_dict[lang_code] = per_class_f1[i]

    # Add zeros for missing classes
    for class_idx in all_classes:
        if class_idx not in present_classes:
            lang_code = inv_map.get(class_idx, f"Unknown_{class_idx}")
            per_class_precision_dict[lang_code] = 0.0
            per_class_recall_dict[lang_code] = 0.0
            per_class_f1_dict[lang_code] = 0.0

    # Create confusion matrix only for present classes
    cm = confusion_matrix(
        true_labels, predictions,
        labels=present_classes
    )

    print(f"Evaluated {total_examples} examples out of {len(dataloader.dataset)} total")

    return {
        'loss': val_loss / len(dataloader),
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'per_class_f1': per_class_f1_dict,
        'per_class_precision': per_class_precision_dict,
        'per_class_recall': per_class_recall_dict,
        'confusion_matrix': cm,
        'present_classes': present_classes,
        'y_true': true_labels,
        'y_pred': predictions
    }


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    # TODO: change for XLM/BERT
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    # tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
    # tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Specify paths to your train and test files

    # Reddit
    train_file = "reddit/reddit_train_dataset.tsv"
    val_file = "reddit/reddit_val_dataset.tsv"
    test_file = "reddit/reddit_test_dataset.tsv"
    # Google
    # train_file = "reddit/reddit_train_google_dataset.tsv"
    # val_file = "reddit/reddit_val_google_dataset.tsv"
    # test_file = "reddit/reddit_test_google_dataset.tsv"

    # Europarl 1k original languages
    # train_file = "europarl_1k_orig/1000_train_dataset.tsv"
    # val_file = "europarl_1k_orig/1000_val_dataset.tsv"
    # test_file = "europarl_1k_orig/1000_test_dataset.tsv"
    # Google
    # train_file = "europarl_1k_orig/1000_train_google_dataset.tsv"
    # val_file = "europarl_1k_orig/1000_val_google_dataset.tsv"
    # test_file = "europarl_1k_orig/1000_test_google_dataset.tsv"
    # Google Cloud
    # train_file = "europarl_1k_orig/1000_train_google_cloud_dataset.tsv"
    # val_file = "europarl_1k_orig/1000_val_google_cloud_dataset.tsv"
    # test_file = "europarl_1k_orig/1000_test_google_cloud_dataset.tsv"

    # Europarl 1k new languages
    # train_file = "europarl_1k_new/1000_train_fi_el_dataset.tsv"
    # val_file = "europarl_1k_new/1000_val_fi_el_dataset.tsv"
    # test_file = "europarl_1k_new/1000_test_fi_el_dataset.tsv"
    # Google
    # train_file = "europarl_1k_new/1000_train_google_fi_el_dataset.tsv"
    # val_file = "europarl_1k_new/1000_val_google_fi_el_dataset.tsv"
    # test_file = "europarl_1k_new/1000_test_google_fi_el_dataset.tsv"


    # languages = ['es', 'fr', 'it', 'de']  # europarl original languages
    # languages = ['el', 'fr', 'fi', 'de']  # europarl new languages
    languages = ['fr', 'ja']  # reddit

    print(f"Training on {train_file}")

    # Prepare train data for training
    train_df = prepare_data(train_file, languages)

    # Prepare validation data for validation
    val_df = prepare_data(val_file, languages)

    # Prepare train dest for testing
    test_df = prepare_data(test_file, languages)

    for lang in languages:
        lang_id = LANGUAGE_MAP[lang]
        print(f"Training examples for {lang}: {(train_df['language'] == lang_id).sum()}")
        print(f"validation examples for {lang}: {(val_df['language'] == lang_id).sum()}")
        print(f"test examples for {lang}: {(test_df['language'] == lang_id).sum()}")

    # Create datasets
    train_dataset = TranslationDataset(
        texts=train_df['text'].values,
        labels=train_df['language'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    val_dataset = TranslationDataset(
        texts=val_df['text'].values,
        labels=val_df['language'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    test_dataset = TranslationDataset(
        texts=test_df['text'].values,
        labels=test_df['language'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE
    )

    # Load model with the frozen model
    model = RobertaWithNonLinearClassifier(
        MODEL_NAME,
        num_labels=NUM_LABELS
    ).to(device)

    # Print a sample batch
    sample_batch = next(iter(train_dataloader))
    print("Input shape:", sample_batch['input_ids'].shape)
    print("Labels:", sample_batch['labels'])
    print("Label distribution:", torch.bincount(sample_batch['labels']))
    print("Label range:", torch.min(sample_batch['labels']).item(),
          "to", torch.max(sample_batch['labels']).item())
    print("Present labels:", sorted(set([int(i) for i in sample_batch['labels']])))

    # Check the model's initial predictions
    model.eval()
    with torch.no_grad():
        sample_outputs = model(
            input_ids=sample_batch['input_ids'].to(device),
            attention_mask=sample_batch['attention_mask'].to(device)
        )
        print("Initial logits:\n", sample_outputs.logits.cpu().numpy())
        print("Initial predictions:", torch.argmax(sample_outputs.logits, dim=1).cpu().numpy())

    # Train the model
    model = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        epochs=EPOCHS
    )

    model_save_path = "trained_models/roberta_reddit_human_translated"
    os.makedirs(model_save_path, exist_ok=True)

    # Save the model state dict
    torch.save(model.state_dict(), os.path.join(model_save_path, "model.pt"))

    # Save the tokenizer
    tokenizer.save_pretrained(model_save_path)

    # Save the configuration
    config_dict = {
        "model_name": MODEL_NAME,
        "training_data": train_file,
        "num_labels": NUM_LABELS,
        "high_learning_rate": HIGH_LEARNING_RATE,
        "low_learning_rate": LOW_LEARNING_RATE,
        "language_map": LANGUAGE_MAP,
        "epochs": EPOCHS
    }
    # Save as JSON for easy loading later
    import json
    with open(os.path.join(model_save_path, "config.json"), "w") as f:
        json.dump(config_dict, f)

    print(f"Model saved to {model_save_path}")

    # Evaluate on test set
    print("\nAfter training, test set metrics:")
    test_metrics = evaluate_model(model, test_dataloader, device)
    print(f"Test Loss: {test_metrics['loss']}")
    print(f"Test Accuracy: {test_metrics['accuracy']}")
    print(f"Test Macro F1: {test_metrics['macro_f1']}")
    print(f"Test Macro Precision: {test_metrics['macro_precision']}")
    print(f"Test Macro Recall: {test_metrics['macro_recall']}")
    print(f"Test Per-class F1: {test_metrics['per_class_f1']}")
    print(f"Test Per-class Precision: {test_metrics['per_class_precision']}")
    print(f"Test Per-class Recall: {test_metrics['per_class_recall']}")

    print("\nTest Confusion Matrix:")
    cm_test = test_metrics['confusion_matrix']
    present_classes_test = test_metrics['present_classes']
    inv_map = {v: k for k, v in LANGUAGE_MAP.items()}
    # Use only the classes present in the data for the confusion matrix
    lang_labels_test = [inv_map.get(idx, f"Unknown_{idx}") for idx in present_classes_test]
    cm_df_test = pd.DataFrame(cm_test, index=lang_labels_test, columns=lang_labels_test)
    print(cm_df_test)


if __name__ == "__main__":
    main()
