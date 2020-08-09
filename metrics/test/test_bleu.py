# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from metrics.bleu import compute_bleu


class BleuTest(tf.test.TestCase):
    def test_compute_bleu_equal(self):
        translation_sentence = [[1, 2, 3, 4, 5, 6, 7]]
        reference_sentence = [[1, 2, 3, 4, 5, 6, 7]]
        bleu = compute_bleu(reference_sentence, translation_sentence)
        actual_bleu = 1.0
        self.assertEqual(bleu, actual_bleu)

    def test_compute_bleu_not_equal(self):
        translation_sentence = [[1, 2, 3, 4]]
        reference_sentence = [[5, 6, 7, 8]]
        bleu = compute_bleu(reference_sentence, translation_sentence)
        actual_bleu = 0.0798679
        self.assertAllClose(bleu, actual_bleu, atol=1e-3)

    def test_compute_multiple_batch(self):
        translation_sentences = [[1, 2, 3, 4], [5, 6, 7, 0]]
        reference_sentences = [[1, 2, 3, 4], [5, 6, 7, 10]]
        bleu = compute_bleu(reference_sentences, translation_sentences)
        actual_bleu = 0.7231
        self.assertAllClose(bleu, actual_bleu, atol=1e-3)

    def test_compute_multiple_ngrams(self):
        reference_sentences = [[1, 2, 1, 13], [12, 6, 7, 4, 8, 9, 10]]
        translation_sentences = [[1, 2, 1, 3], [5, 6, 7, 4]]
        bleu = compute_bleu(reference_sentences, translation_sentences)
        actual_bleu = 0.3436
        self.assertAllClose(bleu, actual_bleu, atol=1e-03)


if __name__ == '__main__':
    tf.test.main()