# ce programme est pour la tache 2 du devoir
import argparse
import re
from typing import List, Tuple
import matplotlib.pyplot as plt
import datasets

DATA = 'Wikipedia_CHARS.txt'
TYPE = 'text'
INTERVALE = 10000


def pre_process(text: str, config: dict) -> List[str]:
    """
    Nettoie et tokenise une chaîne de texte en fonction d'un dictionnaire de configuration.
    """
    text = re.sub('\n', '', text)
    text = re.sub('\t', '', text)

    if config.get('lowercase'):
        text = text.lower()
    if config.get('replace_numbers'):
        text = re.sub(r'\d+', '_NUM_', text)

    tokens = []
    if config.get('segmentation') == 'space':
        tokens = text.split()
    elif config.get('segmentation') == 'punct':
        tokens = re.findall(r'\w+|[^\s\w]', text)
    elif config.get('segmentation') == 'both':
        text_no_punct = re.sub(r'[^\w\s\']', ' ', text)
        tokens = text_no_punct.split()
    return tokens


def generer_donnees(dataset, config, checkpoint_interval: int) -> List[Tuple[int, int]]:
    """
    Génère les points de données (x, y) pour la courbe de croissance du vocabulaire.
    """

    def _generate_tokens_for_plot(data, args_config):
        for example in data:
            yield from pre_process(example['text'], args_config)

    token_generator = _generate_tokens_for_plot(dataset, config)

    print(f"Mode de découpage: {config.get('segmentation')}...")
    seen_words = set()
    data_points = [(0, 0)] # (total token, vocabulary size)
    total_tokens = 0

    for token in token_generator:
        total_tokens += 1
        seen_words.add(token)

        if total_tokens % checkpoint_interval == 0:
            current_vocabulary_size = len(seen_words)
            data_points.append((total_tokens, current_vocabulary_size))

    data_points.append((total_tokens, len(seen_words)))
    return data_points


def tracer_courbe(dataset):
    """
    Exécute la logique pour générer et sauvegarder le graphique de croissance du vocabulaire.
    """
    configurations = [
        {'segmentation': 'space', 'lowercase': True, 'replace_numbers': True},
        {'segmentation': 'punct', 'lowercase': True, 'replace_numbers': True},
        {'segmentation': 'both', 'lowercase': True, 'replace_numbers': True},
    ]

    plt.figure(figsize=(12, 8))

    for config in configurations:
        # Renommé la fonction pour plus de clarté
        data = generer_donnees(dataset, config, INTERVALE)
        x_values, y_values = zip(*data)
        plt.plot(x_values, y_values, label=f"Segmentation: '{config.get('segmentation')}'")

    plt.title(f'Croissance du vocabulaire (Types vs. Tokens) - {len(dataset)} examples', fontsize=16)
    plt.xlabel('Nombre de tokens traités (en millions)', fontsize=12)
    # 0.5 sur le graphique -> 500 000 tokens
    # 1.0 signifie 1 000 000 (1 million) de tokens.
    # 1.5 signifie 1 500 000 (1.5 million) de tokens, et ainsi de suite.
    plt.ylabel('Taille du vocabulaire (Nombre de types uniques)', fontsize=12)
    plt.legend()
    plt.grid(True)

    filename = "vocabulary_growth_comparison.png"
    plt.savefig(filename)
    plt.show()


def main() -> None:
    """
    Fonction principale pour analyser les arguments, traiter le jeu de données et tracer la courbe.
    """
    parser = argparse.ArgumentParser(
        description="Génère une courbe de croissance du vocabulaire pour le jeu de données Wikipedia."
    )
    parser.add_argument(
        "-n", "--num_examples", type=int, default=None,
    )

    args = parser.parse_args()

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

    tracer_courbe(dataset_subset)
    print("terminer la courbe...")


if __name__ == "__main__":
    main()

