# ce programme est pour la tache 1 du devoir
import argparse, sys, re, string, time
from typing import List, Tuple
import matplotlib.pyplot as plt

import datasets
import nltk
from collections import Counter
import numpy as np
import bpe



"""
count.py must produce the vocabulary (the number of
different words) and the frequency of these words in the read part
of the dataset, but must also display the number of examples in the
dataset considered, the time taken to count the words once the word
segmentation has been performed, the type of segmentation performed,
the size of the vocabulary (number of different words), and the
total time in seconds to perform this processing

Counts Word Frequencies: The core function is to read text from the dataset and count how many times each word appears.
Offers Preprocessing Options: It must allow the user to experiment with different text cleaning and tokenization strategies from the command line, such as:

    Converting all text to lowercase.
    
    Replacing numbers with a special symbol (like @).

    Changing how the text is split into words (simple spaces vs. spaces and punctuation).
    
Provides a Detailed Report: After processing, the script must print two things:

    A list of all the unique words found and their frequencies, sorted from most common to least.
    
"""


DATA = 'Wikipedia_CHARS.txt'
TYPE = 'text'

def main() -> None:

    """
        Main function to parse arguments, process dataset, and count words.
        """
    #Argument Parsing
    parser = argparse.ArgumentParser(
        description="Count word frequencies in the BioMedTok/Wikipedia dataset with various preprocessing options."
    )
    parser.add_argument(
        "-n", "--num_examples", type=int, default=None,
    )

    # par d'effault, on va lowercase
    parser.add_argument(
        "-l", "--lowercase",
        action="store_false",
        help="Convert text to lowercase before tokenizing."
    )

    # par défaut, on remplace les chiffres par '_NUM_'.
    parser.add_argument(
        "-d", "--replace_numbers",
        action="store_false",
        help="Replace all sequences of digits with '_NUM_' ."
    )
    parser.add_argument(
        "-s", "--segmentation",
        choices=['space', 'punct', 'both'],
        default='both',
        help="Tokenization strategy: 'space' splits only on whitespace, 'punct' splits on whitespace and punctuation."
    )

    args = parser.parse_args()
    print(f'voici les arguments: {args}')

    total_start_time = time.time()

    ds = datasets.load_dataset(TYPE, data_files=DATA)
    ds_train = ds.get('train')
    assert ds_train is not None

    dataset_size = ds_train.num_rows

    if args.num_examples is None:
        num_to_process = dataset_size
    else:
        # If user specified -n, use their number but validate it against the dataset size
        num_to_process = args.num_examples
        if num_to_process > dataset_size:
            print(f"Warning: Requested {num_to_process} examples, but the dataset only has {dataset_size}...")
            print(f"Processing all {dataset_size} examples instead.")
            num_to_process = dataset_size

    dataset_subset = ds_train.select(range(num_to_process))
    assert type(dataset_subset) == datasets.Dataset

    def _generate_tokens(dataset: datasets.arrow_dataset.Dataset, args: argparse.Namespace):
        for example in dataset:
            yield from pre_process(example['text'], args)

    token_generator = _generate_tokens(dataset_subset, args)

    count_start_time = time.time()
    word_counts = Counter(token_generator)
    count_time = time.time() - count_start_time

    for word, count in word_counts.most_common(50):  # Print top 50
        print(f"{word} {count}")

    # least_common_words = word_counts.most_common()[:] # Print least - reversed order
    # for word, count in reversed(least_common_words):
    #     print(f"{word} {count}")

    for byte, count in word_counts:
        bpe.learn_bpe(byte, 3)

    total_time = time.time() - total_start_time

    total_tokens = sum(word_counts.values())
    vocabulary_size = len(word_counts)

    # resume
    print("\n" + "=" * 50)
    print(f"Summary: Processed {num_to_process} examples in {total_time:.2f}s.")
    print(f"Vocabulary Size (Types): {vocabulary_size}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Counting time: {count_time:.2f}s")
    print("=" * 50)

def pre_process(text: str, args: argparse.Namespace) -> List[str]:
    """
    Nettoie et tokenise une chaîne de texte en fonction des arguments de ligne de commande.
    """
    text = re.sub('\n', '', text)
    text = re.sub('\t', '', text)
    # text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    if args.replace_numbers:
        text = re.sub(r'\d+', '_NUM_', text)

    if args.lowercase:
        text = text.lower()

    tokens = []
    # tokens = nltk.word_tokenize(text)
    if args.segmentation == 'space':
        # Simple split on any whitespace
        # "Alo mon ami, comment vas-tu?" -> ['Alo', 'mon', 'ami,', 'comment', 'vas-tu?']
        # "L'avion, en 1919, décolle." -> ["l'avion,", 'en', '1919,', 'décolle.']
        tokens = text.split()
    elif args.segmentation == 'punct':
        # garde la ponctuation pour les decouper
        # Ex: "Alo mon ami, comment vas-tu?" -> ['Alo', 'mon', 'ami', ',', 'comment', 'vas', '-', 'tu', '?']
        # "L'avion, en 1919, décolle." -> ['L', "'", 'avion', ',', 'en', '1919', ',', 'décolle', '.']
        tokens = re.findall(r'\w+|[^\s\w]', text)
    elif args.segmentation == 'both':
        # enleve la ponctuation et coupe sur espace
        # !,";.... n'est pas considere comme token
        # garde toutefois les apostrophes francais comme: "l'afghanistan"
        # ex: "Alo mon ami, comment vas-tu?" -> ['Alo', 'mon', 'ami', 'comment', 'vas', 'tu']
        # "L'avion, en 1919, décolle." -> ["l'avion", 'en', '1919', 'décolle']
        text_no_punct = re.sub(r'[^\w\s\']', ' ', text)
        tokens = text_no_punct.split()


    return tokens


if __name__ == "__main__":
    # ne remplace pas les chiffres par des charactere speciaux
    # python count.py -n 1000 -s space -> Vocabulary Size (Types): 202486, Total Tokens: 2660258
    # python count.py -n 1000 -s punct -> Vocabulary Size (Types): 98640, Total Tokens: 3405148
    # python count.py -n 1000 -> Vocabulary Size (Types): 106512, Total Tokens: 2661363

    # remplace les chiffre par des charactere speciaux
    main()
