import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from googletrans import Translator
import time
import html
import re
import csv
# for Reddit
import emoji
# # for Google Cloud Translation API
# from google.cloud import translate_v2 as translate
# import os
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../../axiomatic-math-454508-q6-2018e61fce04.json"
# for calculate_tokens
from transformers import RobertaTokenizer


# create tatoeba TSV files containing 800 sentences with at least 30 words each
def filter_tatoeba_tsv(input_paths, output_paths):
    # Filtering tsv files
    for input_path, output_path in zip(input_paths, output_paths):
        # Read the TSV file (assuming it's tab-separated)
        df = pd.read_csv(input_path, delimiter="\t", header=None, names=["id", "original", "otherCol", "translated"],
                         on_bad_lines='skip')

        # Filter rows where the translated sentence has 30 or more words
        filtered_df = df[df["translated"].apply(count_words) >= 30]

        # Randomly sample 800 pairs (if more exist)
        if len(filtered_df) > 800:
            filtered_df = filtered_df.sample(n=800, random_state=42)

        # Keep only the original and translated sentence columns
        filtered_pairs_df = filtered_df[["original", "translated"]]

        # Save the filtered pairs to a new TSV file
        filtered_pairs_df.to_csv(output_path, sep="\t", index=False, header=False)

        print(f"Dataset saved to: {output_path}")


# old create_dataset - only train and val (called test) 80%- train, 20%- validation
def create_dataset_no_test(file_paths, language_labels, train_dataset_path, test_dataset_path):

    test_data = []
    train_data = []
    for file_path, origin_language in zip(file_paths, language_labels):
        data = pd.read_csv(file_path, sep='\t', header=None, names=['original', 'translated'])
        data['origin_language'] = origin_language
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        train_data.append(train)
        test_data.append(test)

    # Concatenate all train and test data
    combined_train_dataset = pd.concat(train_data, ignore_index=True)
    combined_test_dataset = pd.concat(test_data, ignore_index=True)

    # Randomize the data
    combined_train_dataset = shuffle(combined_train_dataset, random_state=42)
    combined_test_dataset = shuffle(combined_test_dataset, random_state=42)

    # Keep only the translated sentence and the origin language
    train_dataset = combined_train_dataset[['translated', 'origin_language']]
    test_dataset = combined_test_dataset[['translated', 'origin_language']]

    # Save the datasets
    train_dataset.to_csv(train_dataset_path, sep='\t', index=False)
    test_dataset.to_csv(test_dataset_path, sep='\t', index=False)


# create dataset- 70%- train, 15%- validation, 15%- test (approximately)
def create_dataset(file_paths, language_labels, train_dataset_path, val_dataset_path, test_dataset_path):
    train_data = []
    val_data = []
    test_data = []

    for file_path, origin_language in zip(file_paths, language_labels):
        data = pd.read_csv(file_path, sep='\t', header=None, names=['original', 'translated'])
        data['origin_language'] = origin_language

        # First split off 15% test set
        train_val, test = train_test_split(data, test_size=0.15, random_state=42)
        # Then split the remaining 85% into 70% train, 15% val
        train, val = train_test_split(train_val, test_size=0.1765, random_state=42)  # 0.1765 * 85% ≈ 15%

        train_data.append(train)
        val_data.append(val)
        test_data.append(test)

    # Combine datasets
    combined_train = pd.concat(train_data, ignore_index=True)
    combined_val = pd.concat(val_data, ignore_index=True)
    combined_test = pd.concat(test_data, ignore_index=True)

    # Shuffle
    combined_train = shuffle(combined_train, random_state=42)
    combined_val = shuffle(combined_val, random_state=42)
    combined_test = shuffle(combined_test, random_state=42)

    # Keep only translated sentence and language label
    combined_train = combined_train[['translated', 'origin_language']]
    combined_val = combined_val[['translated', 'origin_language']]
    combined_test = combined_test[['translated', 'origin_language']]

    # Save to files
    combined_train.to_csv(train_dataset_path, sep='\t', index=False)
    combined_val.to_csv(val_dataset_path, sep='\t', index=False)
    combined_test.to_csv(test_dataset_path, sep='\t', index=False)


# using the free version of Google Translate (googletrans)
def google_translate_files(input_paths, output_paths, languages):
    for input_path, output_path, src_lang in zip(input_paths, output_paths, languages):
        df = pd.read_csv(input_path, sep="\t", header=None, names=["original", "translated"])
        translator = Translator()

        translated_texts = []
        for i, sentence in enumerate(df["original"]):
            try:
                # Attempt full sentence translation first
                translated = translator.translate(sentence, src=src_lang, dest='en')
                translated_text = html.unescape(translated.text)
                translated_clean = re.sub(r'([.?!])(?=[A-Z])', r'\1 ', translated_text)
                translated_texts.append(translated_clean)
                time.sleep(0.5)

            except Exception as e:
                print(f"[Full translation failed at row {i}]: {e}")
                try:
                    # Split sentence on punctuation + space
                    chunks = re.split(r'(?<=[.!?])\s+', sentence)
                    chunk_translations = []
                    for chunk in chunks:
                        if chunk.strip():
                            chunk_translated = translator.translate(chunk, src=src_lang, dest='en')
                            chunk_text = html.unescape(chunk_translated.text)
                            chunk_translations.append(chunk_text)
                            time.sleep(0.5)  # Sleep per chunk to prevent rate limiting
                    combined = ' '.join(chunk_translations)
                    combined_clean = re.sub(r'([.?!])(?=[A-Z])', r'\1 ', combined)
                    translated_texts.append(combined_clean)
                except Exception as e2:
                    print(f"[Fallback translation also failed at row {i}]: {e2}")
                    translated_texts.append("")

        df["google_translation"] = translated_texts
        df[["original", "google_translation"]].to_csv(output_path, sep="\t", index=False, header=False)
        print(f"Saved translated file to: {output_path}")


# using Google Cloud Translation API
def google_cloud_translate_files(input_paths, output_paths, languages):
    for input_path, output_path, src_lang in zip(input_paths, output_paths, languages):
        # Load the file
        df = pd.read_csv(input_path, sep="\t", header=None, names=["original", "translated"])

        # Initialize the translator
        translate_client = translate.Client()
        translated_texts = []

        for i, text in enumerate(df["original"]):
            if pd.isna(text) or not isinstance(text, str):
                translated_texts.append("")
                continue
            try:
                result = translate_client.translate(text, source_language=src_lang, target_language="en")
                translated_text = html.unescape(result["translatedText"])
                translated_clean = re.sub(r'([.?!])(?=[A-Z])', r'\1 ', translated_text)
                translated_texts.append(translated_clean)
            except Exception as e:
                print(f"Error at row {i}: {e}")
                translated_texts.append("")
            time.sleep(0.1)

        # Save the results
        df["google_translation"] = translated_texts
        df[["original", "google_translation"]].to_csv(output_path, sep="\t", index=False, header=False)
        print(f"Saved to {output_path}")


# create a TSV file for each language of pairs of origin text and translated text
def create_europarl_files(original_files, translated_files, output_files):
    for original_file, translated_file, output_file in zip(original_files, translated_files, output_files):
        # Read the original and translated text files
        with open(original_file, 'r', encoding='utf-8') as f1, open(translated_file, 'r', encoding='utf-8') as f2:
            original_sentences = f1.readlines()
            translated_sentences = f2.readlines()

        # Initialize variables to store valid pairs
        valid_pairs = []

        # Variables to accumulate paragraph lines
        original_paragraph = ''
        translated_paragraph = ''

        # Process the sentences in pairs
        for orig, trans in zip(original_sentences, translated_sentences):
            # Decode HTML entities to handle things like &quot; and &#39;
            orig = html.unescape(orig.strip())
            trans = html.unescape(trans.strip())

            # Append the current line to the paragraph
            original_paragraph += orig + ' '
            translated_paragraph += trans + ' '

            # Check if the current paragraph has at least 500 words
            if count_words(original_paragraph) >= 500 and count_words(translated_paragraph) >= 500:
                valid_pairs.append((original_paragraph.strip(), translated_paragraph.strip()))

                # Reset for the next paragraph
                original_paragraph = ''
                translated_paragraph = ''

            # Stop when we have 1000 valid pairs
            if len(valid_pairs) >= 1000:
                break

        # Write valid pairs to the TSV file
        with open(output_file, 'w', newline='', encoding='utf-8') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            for pair in valid_pairs:
                writer.writerow(pair)


# Reddit

def contains_emoji(text):
    # Regex to match basic emoticons (e.g., :), ;), :-P)
    txt_emoji_regex = re.compile(r'(8|:|;|=)(\^|\'|-)?(\)|\(|D|P|p)')
    return bool(txt_emoji_regex.search(text)) or bool(emoji.emoji_list(text))  # returns list of emoji matches


# create a TSV file for each language of pairs of origin text and translated text. Only take texts without emojis.
def filter_reddit_tsv(input_paths, output_paths, n):
    # Process each input file and save the result to the corresponding output file
    for input_path, output_path in zip(input_paths, output_paths):
        valid_pairs = []

        # Read and clean the input file
        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            cleaned_lines = (line.replace('\x00', '') for line in f)
            reader = csv.reader(cleaned_lines, delimiter="\t")
            for row in reader:
                if len(row) != 3:
                    continue
                _, origin_text, translated_text = row
                # Skip rows with newlines in origin or translated text
                if "\n" in origin_text or "\n" in translated_text:
                    continue
                # Check if there are no emojis in the texts
                if not contains_emoji(origin_text) and not contains_emoji(translated_text):
                    valid_pairs.append((origin_text, translated_text))
                    # Stop if we have reached the desired number of valid pairs
                    if len(valid_pairs) >= n:
                        break

        # Write the valid pairs to the corresponding output file
        with open(output_path, "w", encoding="utf-8") as out:
            for origin, translated in valid_pairs:
                out.write(f"{origin}\t{translated}\n")


def count_words_and_sentences(files):
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Unescape HTML entities and strip whitespace
        sentences = [html.unescape(line.strip()) for line in lines if line.strip()]

        num_sentences = len(sentences)
        num_words = sum(len(sentence.split()) for sentence in sentences)

        print(f"{file}:")
        print(f"  Sentences: {num_sentences}")
        print(f"  Words: {num_words}\n")


def trim_tsv(input_paths, output_paths, n=1000):
    for input_path, output_path in zip(input_paths, output_paths):
        # Load the TSV file (assumes 2 columns, no header)
        df = pd.read_csv(input_path, sep="\t", header=None)

        # Take the first `n` rows
        trimmed_df = df.iloc[n:]

        # Save to output file
        trimmed_df.to_csv(output_path, sep="\t", index=False, header=False)
        print(f"Saved first {n} rows from {input_path} to {output_path}")


def concatenate_tsv_files(input_paths, output_path):
    all_data = []
    for path in input_paths:
        df = pd.read_csv(path, sep="\t", header=None)
        if df.shape[1] != 2:
            raise ValueError(f"File {path} does not have exactly two columns.")
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv(output_path, sep="\t", index=False, header=False)
    print(f"Saved concatenated file to: {output_path}")


# used to calculate the number of tokens (used to make sure the sentences aren't too long for RoBERTa)
def calculate_tokens():
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Load your dataset (assuming it's a TSV file with "text" column)
    df_train = pd.read_csv("../../data/train_dataset.tsv", sep='\t')

    # Tokenize all sentences
    token_lens_train = [len(tokenizer.tokenize(sent)) for sent in df_train["translated"]]

    # Print statistics
    print(f"Max tokens train: {max(token_lens_train)}")
    print(f"Average tokens train: {sum(token_lens_train) / len(token_lens_train):.2f}")

    # Load your dataset (assuming it's a TSV file with "text" column)
    df_test = pd.read_csv("../../data/test_dataset.tsv", sep='\t')

    # Tokenize all sentences
    token_lens_test = [len(tokenizer.tokenize(sent)) for sent in df_test["translated"]]

    # Print statistics
    print(f"Max tokens test: {max(token_lens_test)}")
    print(f"Average tokens test: {sum(token_lens_test) / len(token_lens_test):.2f}")


if __name__ == '__main__':

    # "fr" for French
    # "de" for German
    # "ru" for Russian
    # "es" for Spanish
    tatoeba_languages = ["fr", "de", "ru", "es"]

    tatoeba_pairs_paths = ["../../data/tatoeba_pairs_French_English.tsv",
                           "../../data/tatoeba_pairs_German_English.tsv",
                           "../../data/tatoeba_pairs_Italian_English.tsv",
                           "../../data/tatoeba_pairs_Spanish_English.tsv"]

    tatoeba_filtered_paths = ["../../data/tatoeba_filtered_French_English_pairs.tsv",
                              "../../data/tatoeba_filtered_German_English_pairs.tsv",
                              "../../data/tatoeba_filtered_Italian_English_pairs.tsv",
                              "../../data/tatoeba_filtered_Spanish_English_pairs.tsv"]

    tatoeba_google_paths = ["../../data/tatoeba_google_French_English_pairs.tsv",
                            "../../data/tatoeba_google_German_English_pairs.tsv",
                            "../../data/tatoeba_google_Italian_English_pairs.tsv",
                            "../../data/tatoeba_google_Spanish_English_pairs.tsv"]

    # Europarl
    europarl_languages = ["fr", "de", "it", "es"]

    europarl_original_paths = ["../../europarl/fr-en/europarl-v7.fr-en.fr",
                               "../../europarl/de-en/europarl-v7.de-en.de",
                               "../../europarl/it-en/europarl-v7.it-en.it",
                               "../../europarl/es-en/europarl-v7.es-en.es"]

    europarl_translated_paths = ["../../europarl/fr-en/europarl-v7.fr-en.en",
                                 "../../europarl/de-en/europarl-v7.de-en.en",
                                 "../../europarl/it-en/europarl-v7.it-en.en",
                                 "../../europarl/es-en/europarl-v7.es-en.en"]

    europarl_1000_pairs_paths = ["../../europarl/1000/1000_fr-en.tsv",
                                 "../../europarl/1000/1000_de-en.tsv",
                                 "../../europarl/1000/1000_it-en.tsv",
                                 "../../europarl/1000/1000_es-en.tsv"]

    europarl_1000_google_pairs_paths = ["../../europarl/1000/1000_google_French_English_pairs.tsv",
                                        "../../europarl/1000/1000_google_German_English_pairs.tsv",
                                        "../../europarl/1000/1000_google_Finnish_English_pairs.tsv",
                                        "../../europarl/1000/1000_google_Spanish_English_pairs.tsv"]

    europarl_1000_google_cloud_pairs_paths = ["../../europarl/1000/1000_google_cloud_French_English_pairs.tsv",
                                              "../../europarl/1000/1000_google_cloud_German_English_pairs.tsv",
                                              "../../europarl/1000/1000_google_cloud_Italian_English_pairs.tsv",
                                              "../../europarl/1000/1000_google_cloud_Spanish_English_pairs.tsv"]

    europarl_new_languages = ["fr", "de", "fi", "el"]

    europarl_fi_el_original_paths = ["../../europarl/fi-en/europarl-v7.fi-en.fi",
                                     "../../europarl/el-en/europarl-v7.el-en.el"]

    europarl_fi_el_translated_paths = ["../../europarl/fi-en/europarl-v7.fi-en.en",
                                       "../../europarl/el-en/europarl-v7.el-en.en"]

    europarl_fi_el_1000_pairs_paths = ["../../europarl/1000/1000_fr-en.tsv",
                                       "../../europarl/1000/1000_de-en.tsv",
                                       "../../europarl/1000/1000_fi-en.tsv",
                                       "../../europarl/1000/1000_el-en.tsv"]

    europarl_fi_el_1000_google_pairs_paths = ["../../europarl/1000/1000_google_French_English_pairs.tsv",
                                              "../../europarl/1000/1000_google_German_English_pairs.tsv",
                                              "../../europarl/1000/1000_google_Finnish_English_pairs.tsv",
                                              "../../europarl/1000/1000_google_Greek_English_pairs.tsv"]

    # Reddit
    reddit_languages = ["ja", "fr"]

    reddit_pairs_paths = ["../../reddit/unified_filtered_ja-en.tsv",
                          "../../reddit/unified_filtered_fr-en.tsv"]

    reddit_google_pairs_paths = ["../../reddit/reddit_google_Japanese_English_pairs.tsv",
                                 "../../reddit/reddit_google_French_English_pairs.tsv"]


    # Tatoeba
    filter_tatoeba_tsv(tatoeba_pairs_paths, tatoeba_filtered_paths)
    create_dataset(tatoeba_filtered_paths, tatoeba_languages,
    "../../data/split/train_tatoeba_dataset.tsv",
                   "../../data/split/val_tatoeba_dataset.tsv",
                   "../../data/split/test_tatoeba_dataset.tsv")
    google_translate_files(tatoeba_filtered_paths, tatoeba_google_paths, tatoeba_languages)
    create_dataset(tatoeba_google_paths, tatoeba_languages,
                   "../../data/split/tatoeba_google_train_dataset.tsv",
                   "../../data/split/tatoeba_google_val_dataset.tsv",
                   "../../data/split/tatoeba_google_test_dataset.tsv")


    # Europarl
    create_europarl_files(europarl_original_paths, europarl_translated_paths, europarl_1000_pairs_paths)
    create_dataset(europarl_1000_pairs_paths, europarl_languages,
                   "../../europarl/1000/split/1000_train_dataset.tsv",
                   "../../europarl/1000/split/1000_val_dataset.tsv",
                   "../../europarl/1000/split/1000_test_dataset.tsv")
    # free google translate (googletrans)
    google_translate_files(europarl_1000_pairs_paths, europarl_1000_google_pairs_paths, europarl_languages)
    create_dataset(europarl_1000_google_pairs_paths, europarl_languages,
                   "../../europarl/1000/split/1000_train_google_dataset.tsv",
                   "../../europarl/1000/split/1000_val_google_dataset.tsv",
                   "../../europarl/1000/split/1000_test_google_dataset.tsv")
    # google cloud API
    google_cloud_translate_files(europarl_1000_pairs_paths, europarl_1000_google_cloud_pairs_paths, europarl_languages)
    create_dataset(europarl_1000_google_cloud_pairs_paths, europarl_languages,
                   "../../europarl/1000/split/1000_train_google_cloud_dataset.tsv",
                   "../../europarl/1000/split/1000_val_google_cloud_dataset.tsv",
                   "../../europarl/1000/split/1000_test_google_cloud_dataset.tsv")

    # europarl new languages
    create_europarl_files(europarl_fi_el_original_paths,
                          europarl_fi_el_translated_paths,
                          ["../../europarl/1000/1000_fi-en.tsv", "../../europarl/1000/1000_el-en.tsv"])

    create_dataset(europarl_fi_el_1000_pairs_paths,
                   europarl_new_languages,
                   "../../europarl/1000/split/1000_train_fi_el_dataset.tsv",
                   "../../europarl/1000/split/1000_val_fi_el_dataset.tsv",
                   "../../europarl/1000/split/1000_test_fi_el_dataset.tsv")
    # free google translate (googletrans)
    google_translate_files(["../../europarl/1000/1000_fi-en.tsv",
                            "../../europarl/1000/1000_el-en.tsv"],
                           ["../../europarl/1000/1000_google_Finnish_English_pairs.tsv",
                            "../../europarl/1000/1000_google_Greek_English_pairs.tsv"],
                           ["fi", "el"])
    create_dataset(europarl_fi_el_1000_google_pairs_paths,
                   europarl_new_languages,
                   "../../europarl/1000/split/1000_train_google_fi_el_dataset.tsv",
                   "../../europarl/1000/split/1000_val_google_fi_el_dataset.tsv",
                   "../../europarl/1000/split/1000_test_google_fi_el_dataset.tsv")


    # Reddit
    # filter train dataset
    filter_reddit_tsv(["../../reddit/train/train_ja-en.tsv", "../../reddit/train/train_fr-en.tsv"],
                      ["../../reddit/train/filtered_train_ja-en.tsv", "../../reddit/train/filtered_train_fr-en.tsv"],
                      3500)
    # filter validation dataset
    filter_reddit_tsv(["../../reddit/valid/valid_ja-en.tsv", "../../reddit/valid/valid_fr-en.tsv"],
                      ["../../reddit/valid/filtered_valid_ja-en.tsv", "../../reddit/valid/filtered_valid_fr-en.tsv"],
                      750)
    # filter test dataset
    filter_reddit_tsv(["../../reddit/test/test_ja-en.tsv", "../../reddit/test/test_fr-en.tsv"],
                      ["../../reddit/test/filtered_test_ja-en.tsv", "../../reddit/test/filtered_test_fr-en.tsv"],
                      750)

    # create a unified file of all Japanese datasets
    concatenate_tsv_files(["../../reddit/test/filtered_test_ja-en.tsv",
                           "../../reddit/train/filtered_train_ja-en.tsv",
                           "../../reddit/valid/filtered_valid_ja-en.tsv"],
                          "../../reddit/unified_filtered_ja-en.tsv")

    # create a unified file of all French datasets
    concatenate_tsv_files(["../../reddit/test/filtered_test_fr-en.tsv",
                           "../../reddit/train/filtered_train_fr-en.tsv",
                           "../../reddit/valid/filtered_valid_fr-en.tsv"],
                          "../../reddit/unified_filtered_fr-en.tsv")

    create_dataset(reddit_pairs_paths,
                   reddit_languages,
                   "../../reddit/train_dataset.tsv",
                   "../../reddit/val_dataset.tsv",
                   "../../reddit/test_dataset.tsv")

    # free google translate (googletrans)
    google_translate_files(reddit_pairs_paths, reddit_google_pairs_paths, reddit_languages)
    create_dataset(reddit_google_pairs_paths,
                   reddit_languages,
                   "../../reddit/train_google_dataset.tsv",
                   "../../reddit/val_google_dataset.tsv",
                   "../../reddit/test_google_dataset.tsv")


