# BPE-tokenization
BPE-Tokenizer-BioMed-Analysis An implementation of the Byte Pair Encoding (BPE) algorithm from scratch, featuring a full text processing pipeline for the BioMedTok Wikipedia dataset. Includes scripts for vocabulary growth visualization (Heaps' Law) and custom tokenization strategies.

# Description: 
### This repository hosts a complete Natural Language Processing (NLP) pipeline designed to analyze biomedical text data and train a subword tokenizer from scratch. Developed as part of an advanced NLP coursework, this project demonstrates the fundamental algorithms behind modern Large Language Models (LLMs) like GPT and RoBERTa.


Key Features:
- From-Scratch BPE Implementation: A pure Python implementation of the Byte Pair Encoding algorithm that iteratively merges frequent character pairs to build a subword vocabulary
- Vocabulary Growth Analysis: A visualization suite using matplotlib to plot Heaps' Law curves, analyzing how token count vs. vocabulary size scales across different segmentation strategies (Space vs. Punctuation)
- High-Performance Text Processing: Optimized scripts (count.py) to process the 357MB BioMedTok dataset, handling regex-based cleaning, lowercasing, and numeric normalization
