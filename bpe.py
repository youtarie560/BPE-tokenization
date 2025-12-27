from collections import defaultdict
from typing import List
import logging

logging.basicConfig(
    level=logging.INFO,
    format='INFO %(asctime)s [bpe.py] ## %(message)s',
    datefmt='%H:%M:%S'
)
# code taken from: https://www.geeksforgeeks.org/nlp/byte-pair-encoding-bpe-in-nlp/
# some little modifications are made
def learn_bpe_devoir(word: str, num_merges: int =3) -> List[int]:
    """
    learn_bpe function is designed to learn and return
    the most frequent character pairs from the input text.
    It also merges these frequent pairs iteratively.

    We iterate through the vocabulaireulary to find the most frequent adjacent character pair
    and perform the merge.
    Process repeats for a defined number of merges (num_merges).
    """
    # au debut, vocabulaire est peuple avec des bytes
    vocabulaire = defaultdict(int)


    for c in word:
        chars = ['<'] + list(c) + ['>']
        for i in range(len(chars) - 1):
            pair = (chars[i], chars[i + 1])
            vocabulaire[pair] += 1
    # vocabulaire va etre peuple avec les subwords

    merges = []
    for _ in range(num_merges):
        if not vocabulaire:
            break

        most_frequent = max(vocabulaire, key=lambda x: vocabulaire[x])
        merges.append(most_frequent)

        new_char = ''.join(most_frequent)
        new_vocabulaire = defaultdict(int)
        for pair in vocabulaire:
            count = vocabulaire[pair]
            if pair == most_frequent:
                continue
            new_pair = list(pair)
            if new_pair[0] == most_frequent[0] and new_pair[1] == most_frequent[1]:
                new_pair[0] = new_char
                new_pair.pop(1)
            new_vocabulaire[tuple(new_pair)] += count
        vocabulaire = new_vocabulaire
    return merges


def learn_bpe(corpus, num_merges=3):
    vocab = defaultdict(int)

    for sentence in corpus.split('.'):
        words = sentence.strip().split()
        for word in words:
            chars = ['<'] + list(word) + ['>']
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                vocab[pair] += 1

    merges = []
    for i in range(num_merges):
        if not vocab:
            break

        most_frequent_pair = max(vocab, key=lambda x: vocab[x])
        merges.append(most_frequent_pair)
        new_token = ''.join(most_frequent_pair)


        # This is the key part that generates the desired output.
        log_message = (
            f"merge {i + 1}: mfp: {most_frequent_pair} new token: "
            f"{new_token}"
        )
        logging.info(log_message)

        new_char = ''.join(most_frequent_pair)
        new_vocab = defaultdict(int)
        for pair in vocab:
            count = vocab[pair]
            if pair == most_frequent_pair:
                continue
            new_pair = list(pair)
            if new_pair[0] == most_frequent_pair[0] and new_pair[1] == most_frequent_pair[1]:
                new_pair[0] = new_char
                new_pair.pop(1)
            new_vocab[tuple(new_pair)] += count
        vocab = new_vocab
    return merges

def apply_bpe(text, merges):
    chars = ['<'] + list(text) + ['>']
    for merge in reversed(merges):
        merged = ''.join(merge)
        new_chars = []
        i = 0
        while i < len(chars) - 1:
            if (chars[i], chars[i + 1]) == merge:
                new_chars.append(merged)
                i += 2
            else:
                new_chars.append(chars[i])
                i += 1
        if i < len(chars):
            new_chars.append(chars[-1])
        chars = new_chars

    final_output = []
    for token in chars:
        if len(token) == 1 and token not in ['<', '>']:
            final_output.append(f"[UNK]: {token}")
        else:
            final_output.append(token)

    return final_output

    return chars

def main() -> None:
    corpus = "hug pug pun bun hugs"

    # Example usage
    new_word = "mug"
    merges = learn_bpe(corpus, num_merges=50000)
    print("Learned Merges:", merges)
    bpe_representation = apply_bpe(new_word, merges)
    print(f"BPE Representation for {new_word}:", bpe_representation)



if __name__ == '__main__':
    main()