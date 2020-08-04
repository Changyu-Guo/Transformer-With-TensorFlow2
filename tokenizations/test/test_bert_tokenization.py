# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import tensorflow as tf
from tokenizations import bert_tokenization


class BertTokenizationTest(tf.test.TestCase):

    def test_full_tokenizer(self):
        vocab_tokens = [
            '[UNK]', '[CLS]', '[SEP]', 'want', '##want',
            '##ed', 'wa', 'un', 'runn', '##ing', ','
        ]
        with tempfile.NamedTemporaryFile(delete=False) as vocab_writer:
            vocab_writer.write(''.join([x + '\n' for x in vocab_tokens]).encode('utf-8'))

            vocab_file = vocab_writer.name

        tokenizer = bert_tokenization.FullTokenizer(vocab_file)
        os.unlink(vocab_file)

        tokens = tokenizer.tokenize(u'UNwant\u00E9d,running')
        self.assertAllEqual(
            tokens,
            ['un', '##want', '##ed', ',', 'runn', '##ing']
        )
        self.assertAllEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [7, 4, 5, 10, 8, 9]
        )

    def test_chinese(self):
        tokenizer = bert_tokenization.BasicTokenizer()

        self.assertAllEqual(
            tokenizer.tokenize(u'ah\u535A\u63A8zz'),
            [u'ah', u'\u535A', u'\u63A8', u'zz']
        )

    def test_basic_tokenizer_lower(self):
        tokenizer = bert_tokenization.BasicTokenizer(do_lower_case=True)
        self.assertAllEqual(
            tokenizer.tokenize(u' \tHeLLo!how  \n Are yoU?  '),
            ['hello', '!', 'how', 'are', 'you', '?']
        )
        self.assertAllEqual(tokenizer.tokenize(u'H\u00E9llo'), ['hello'])

    def test_basic_tokenizer_no_split_on_punc(self):
        tokenizer = bert_tokenization.BasicTokenizer(
            do_lower_case=True, split_on_punc=False)

        self.assertAllEqual(
            tokenizer.tokenize(u" \tHeLLo!how  \n Are yoU?  "),
            ["hello!how", "are", "you?"])

    def test_wordpiece_tokenizer(self):
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
            "##ing", "##!", "!"
        ]

        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = bert_tokenization.WordpieceTokenizer(vocab=vocab)

        self.assertAllEqual(tokenizer.tokenize(""), [])

        self.assertAllEqual(
            tokenizer.tokenize("unwanted running"),
            ["un", "##want", "##ed", "runn", "##ing"])

        self.assertAllEqual(
            tokenizer.tokenize("unwanted running !"),
            ["un", "##want", "##ed", "runn", "##ing", "!"])

        self.assertAllEqual(
            tokenizer.tokenize("unwanted running!"),
            ["un", "##want", "##ed", "runn", "##ing", "##!"])

        self.assertAllEqual(
            tokenizer.tokenize("unwantedX running"), ["[UNK]", "runn", "##ing"])

    def test_convert_tokens_to_ids(self):
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
            "##ing"
        ]

        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i

        self.assertAllEqual(
            bert_tokenization.convert_tokens_to_ids(
                vocab,
                ["un", "##want", "##ed", "runn", "##ing"]
            ),
            [7, 4, 5, 8, 9]
        )

    def test_is_whitespace(self):
        self.assertTrue(bert_tokenization._is_whitespace(u" "))
        self.assertTrue(bert_tokenization._is_whitespace(u"\t"))
        self.assertTrue(bert_tokenization._is_whitespace(u"\r"))
        self.assertTrue(bert_tokenization._is_whitespace(u"\n"))
        self.assertTrue(bert_tokenization._is_whitespace(u"\u00A0"))

        self.assertFalse(bert_tokenization._is_whitespace(u"A"))
        self.assertFalse(bert_tokenization._is_whitespace(u"-"))

    def test_is_control(self):
        self.assertTrue(bert_tokenization._is_control(u"\u0005"))

        self.assertFalse(bert_tokenization._is_control(u"A"))
        self.assertFalse(bert_tokenization._is_control(u" "))
        self.assertFalse(bert_tokenization._is_control(u"\t"))
        self.assertFalse(bert_tokenization._is_control(u"\r"))
        self.assertFalse(bert_tokenization._is_control(u"\U0001F4A9"))

    def test_is_punctuation(self):
        self.assertTrue(bert_tokenization._is_punctuation(u"-"))
        self.assertTrue(bert_tokenization._is_punctuation(u"$"))
        self.assertTrue(bert_tokenization._is_punctuation(u"`"))
        self.assertTrue(bert_tokenization._is_punctuation(u"."))

        self.assertFalse(bert_tokenization._is_punctuation(u"A"))
        self.assertFalse(bert_tokenization._is_punctuation(u" "))


if __name__ == '__main__':
    tf.test.main()
