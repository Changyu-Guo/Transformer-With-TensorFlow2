# -*- coding: utf - 8 -*-

import sentencepiece as spm


spm.SentencePieceTrainer.Train('--input=corpus.txt --model_prefix=m --vocab_size=332')

sp = spm.SentencePieceProcessor()
sp.load('m.model')

print(sp.EncodeAsPieces('This is a test'))
print(sp.EncodeAsIds('This is a test'))
