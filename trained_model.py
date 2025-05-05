import torch
import os
import json
import main
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader


if __name__ == "__main__":

    os.environ["USE_TF"] = "0"  # Explicitly tell Transformers not to use TensorFlow

    model_save_path = "../../trained_models/roberta_reddit_google"  # TODO: change

    with open(os.path.join(model_save_path, "config.json"), "r") as f:
        config_dict = json.load(f)

    tokenizer = RobertaTokenizer.from_pretrained(model_save_path)

    model = main.RobertaWithNonLinearClassifier(
        pretrained_model_name=config_dict["model_name"],
        num_labels=config_dict["num_labels"],
    )

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(os.path.join(model_save_path, "model.pt")))
    else:
        model.load_state_dict(torch.load(os.path.join(model_save_path, "model.pt"), map_location=torch.device('cpu')))

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Model and tokenizer successfully loaded!")

    languages = list(config_dict["language_map"].keys())
    test_file = "../../reddit/test_dataset.tsv"  # Evaluation data
    test_df = main.prepare_data(test_file, languages, config_dict["language_map"])
    test_dataset = main.TranslationDataset(
            texts=test_df['text'].values,
            labels=test_df['language'].values,
            tokenizer=tokenizer,
            max_len=main.MAX_LEN
        )

    test_dataloader = DataLoader(
            test_dataset,
            batch_size=main.BATCH_SIZE
        )

    print("\nAfter training, test set metrics:")
    test_metrics = main.evaluate_model(model, test_dataloader, device, len(languages), config_dict["language_map"])
    print(f"Test Loss: {test_metrics['loss']}")
    print(f"Test Accuracy: {test_metrics['accuracy']}")
    print(f"Test Macro F1: {test_metrics['macro_f1']}")
    print(f"Test Macro Precision: {test_metrics['macro_precision']}")
    print(f"Test Macro Recall: {test_metrics['macro_recall']}")
    print(f"Test Per-class F1: {test_metrics['per_class_f1']}")
    print(f"Test Per-class Precision: {test_metrics['per_class_precision']}")
    print(f"Test Per-class Recall: {test_metrics['per_class_recall']}")
