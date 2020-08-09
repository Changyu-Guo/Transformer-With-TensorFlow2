# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import collections
import numpy as np
import tensorflow as tf


def _get_ngrams(sentence, max_n):
    ngram_counts = collections.Counter()

    # 1-gram ... n-gram
    for n in range(1, max_n + 1):
        # slide window
        for i in range(0, len(sentence) - n + 1):
            ngram = tuple(sentence[i: i + n])
            ngram_counts[ngram] += 1

    return ngram_counts


def compute_bleu(
        reference_sentences,
        translation_sentences,
        max_n=4,
        use_bp=True
):
    reference_length = 0
    translation_length = 0
    bp = 1.0
    geo_mean = 0

    matches_by_n = [0] * max_n
    possible_matches_by_n = [0] * max_n

    for (reference, translation) in zip(reference_sentences, translation_sentences):
        reference_ngram_counts = _get_ngrams(reference, max_n)
        translation_ngram_counts = _get_ngrams(translation, max_n)

        overlap = dict((
            ngram,
            min(count, translation_ngram_counts[ngram])
        ) for ngram, count in reference_ngram_counts.items())

        for ngram in overlap:
            matches_by_n[len(ngram) - 1] += overlap[ngram]
        for ngram in translation_ngram_counts:
            possible_matches_by_n[len(ngram) - 1] += translation_ngram_counts[ngram]

    precisions = [0] * max_n
    smooth = 1.0

    for i in range(0, max_n):
        if possible_matches_by_n[i] > 0:
            if matches_by_n[i] > 0:
                precisions[i] = matches_by_n[i] / possible_matches_by_n[i]
            else:
                smooth *= 2
                precisions[i] = 1.0 / (smooth * possible_matches_by_n[i])
        else:
            precisions[i] = 0.0

    if max(precisions) > 0:
        p_log_sum = sum(math.log(p) for p in precisions if p)
        geo_mean = math.exp(p_log_sum / max_n)

    if use_bp:
        if not reference_length:
            bp = 1.0
        else:
            ratio = translation_length / reference_length
            if ratio <= 0.0:
                bp = 0.0
            elif ratio >= 1.0:
                bp = 1.0
            else:
                bp = math.exp(1 - 1. / ratio)
    bleu = geo_mean * bp
    print(bleu)
    return np.float32(bleu)


def bleu_score(logits, labels):
    predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    bleu = tf.py_function(compute_bleu, (labels, predictions), tf.float32)
    return bleu, tf.constant(1.0)