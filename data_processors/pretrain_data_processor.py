# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import random

import tensorflow as tf
import tokenization
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT modeling was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_mask", False,
    "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_integer(
    "max_ngram_size", None,
    "Mask contiguous whole words (n-grams) of up to `max_ngram_size` using a "
    "weighting scheme to favor shorter n-grams. "
    "Note: `--do_whole_word_mask=True` must also be set when n-gram masking.")

flags.DEFINE_bool(
    "gzip_compress", False,
    "Whether to use `GZIP` compress option to get compressed TFRecord files.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")


class TrainingInstance:
    def __init__(
            self,
            tokens,  # shape: [str], 当前样本的 WordPiece tokens (一段文本经过 Word Piece 分词后得到的 tokens)
            segment_ids,  # shape: [int], 用于区分当前 token 属于句子 A 还是 B,  0 -> A, 1 -> B
            masked_lm_positions,  # shape: [int], 当前样本中被 mask 的词的位置
            masked_lm_labels,  # shape: [str], 当前样本中被 mask 的词的真实标签
            is_random_next  # shape: True or False, 句子 B 是不是一个随机的句子
    ):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
        self.is_random_next = is_random_next

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_files(
        instances,
        tokenizer,
        max_seq_length,
        max_predictions_per_seq,
        output_files,
        gzip_compress
):
    """
    :param instances: 所有的样本
    :param tokenizer: 用于将 tokens 和 masked_lm_labels 转换为 id
    :param max_seq_length: 样本的最大句长. 最后的样本还是都 padding 成了相同的长度，但未 padding 时长度是随机的
    :param max_predictions_per_seq: 最大 mask 词数，用于将样本中的 mask tokens padding 成相同长度
    :param output_files: tfrecord 保存的文件
    :param gzip_compress: 是否启用 gzip 压缩
    :return:
    """
    writers = []
    for output_file in output_files:
        writers.append(
            tf.io.TFRecordWriter(
                output_file, options='GZIP' if gzip_compress else ''
            )
        )

    writer_index = 0
    total_written = 0
    # 1. 将 tokens 转换为 ids
    # 2. 将长度对齐
    # 3. 生成 Feature 和 Example
    # 4. 每文件一条，将数据交替写入文件
    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        next_sentence_label = 1 if instance.is_random_next else 0

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
        features["next_sentence_labels"] = create_int_feature([next_sentence_label])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)
        total_written += 1

    for writer in writers:
        writer.close()

    logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_training_instances(
        input_files,
        tokenizer,
        max_seq_length,
        dupe_factor,
        short_seq_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        rng,
        do_whole_word_mask=False,
        max_ngram_size=None
):
    """
    :param input_files: 用于预训练的所有文件
    :param tokenizer: tokenizer
    :param max_seq_length: 单个样本的最大长度
    :param dupe_factor: 预训练文本重复次数，因为在构造样本的时候加入了很多随机因此，因此可以使用原始文件重复生成不同样本
    :param short_seq_prob: 短样本比例
    :param masked_lm_prob: 样本中 mask 的词比例
    :param max_predictions_per_seq: 每个样本最多被 mask 掉多少词，与 masked_lm_prob 共同计算出要 mask 掉多少词
    :param rng: random
    :param do_whole_word_mask: 是否全次遮盖，和 max_gram_size 配合使用
    :param max_ngram_size: 最多可以多少个词连在一起被 mask，如果不指定，就默认 mask 掉一个词（注意是词不是 token）
    :return:
    """
    all_documents = [[]]

    for input_file in input_files:
        with tf.io.gfile.GFile(input_file, 'rb') as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())

                # 读到文件末尾
                if not line:
                    break

                # 读到空行
                # 说明一个文档读完
                line = line.strip()
                if not line:
                    all_documents.append([])

                # 将当前文档的 tokens 加入
                # 注意是用的 append 而不是 extend
                # 因此 all_documents 的结构为 [[[], []], [[], []], [[]]]
                # 一个文档中每一行是一个列表
                tokens = tokenizer.tokenize(line)  # WordPiece Tokenize
                if tokens:
                    all_documents[-1].append(tokens)

    # Remove empty documents
    all_documents = [x for x in all_documents if x]

    # 将文档打乱
    rng.shuffle(all_documents)

    # 构造 masked lm 时会用到
    vocab_words = list(tokenizer.vocab.keys())

    # 创建训练样本
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(
                    all_documents,
                    document_index,
                    max_seq_length,
                    short_seq_prob,
                    masked_lm_prob,
                    max_predictions_per_seq,
                    vocab_words,
                    rng,
                    do_whole_word_mask,
                    max_ngram_size
                )
            )
    rng.shuffle(instances)
    return instances


def create_instances_from_document(
        all_documents,
        document_index,
        max_seq_length,
        short_seq_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        vocab_words,
        rng,
        do_whole_word_mask=False,
        max_ngram_size=None
):
    document = all_documents[document_index]
    max_num_tokens = max_seq_length - 3  # -3 是因为有 [CLS], [SEP], [SEP]

    # short_seq_prob 为长度小于 max_seq_length 的样本比例
    # 这里是以一定概率将句子的总长度截短
    # 因为在 fine-tune 的过程中 max_seq_length 可能和 pre train 的时候不同
    # 因此这里这样做可以防止过拟合
    # 注意：这里只是改变了句子的有效长度，在训练的过程中，真实长度还是 max_seq_length
    # 只不过如果总长度被截断，后面会 padding 的更长
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    instances = []

    # 对于每一篇文章，维护一个 current_chunk
    # 后面不断往 current_chunk 中添加 segment 中的 token
    # 至到将 segment 添加完或者 current_chunk 达到指定长度(target_seq_length)
    current_chunk = []
    current_length = 0
    i = 0

    while i < len(document):
        # 一次添加一句进来
        # 为了保证在做 NSP 任务的时候句子连贯
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)

        # 已经将所有 segment 添加进 current_chunk 或者 current_chunk 的长度达到指定值
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # 从 current_chunk 中随机选择一个切分点
                # 将 document 切分为 A 和 B 两个句子
                # 注意 A B 可能包含原始文档中的多条句子
                a_end = 1  # A 结束的位置，不包括 1，默认 A 有一条句子
                if len(current_chunk) >= 2:
                    # 注意 randint 是包含上限的
                    # 因此 B 中至少会包含一条句子
                    a_end = rng.randint(1, len(current_chunk) - 1)

                # 构造 A
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                # 对于 B，判断是否是随机的一条句子
                # 如果 current_chunk 中只有一条句子
                # 那么这条句子肯定会分配给 A
                # 因此 B 一定是一条随机的句子
                # 另外 B 也有 0.5 的几率是随机的
                tokens_b = []
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True

                    # 确定还有多少 tokens 是留给 B 的
                    # 以从随机文件中挑选随机字符
                    target_b_length = target_seq_length - len(tokens_a)

                    # 随机选择一篇文章
                    # 如果 10 次都选择出当前文章
                    # 说明当前文章命中注定，就选它叭
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    # 从 random document 中选一段连续的 segment 加入 B 中
                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break

                    # 没有用到的不要仍
                    # 留下来构造下一个样本
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments

                    # B 使用真实的文本
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                # 如果能将 A B 的总长度控制在 target_seq_length 以内就尽量控制
                # 如果超过了则就只截断到 max_num_tokens (保证句子完整性，有利于做 NSP 任务)
                # 另外，这样也可以增加长度的随机性
                # 因为超过了 target_seq_length 不一定超过 max_num_tokens
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append('[CLS]')
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append('[SEP]')
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append('[SEP]')
                segment_ids.append(1)

                (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng, do_whole_word_mask,
                    max_ngram_size
                )
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels
                )
                instances.append(instance)

            # 重置 chunk
            # 以便构造下一个样本
            current_chunk = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple(
    'MaskedLmInstance',
    ['index', 'label']
)

# 记录 gram 在 WordPiece tokens 中的位置
# 前开后闭
# 例如：
#   words:  ['The', 'doghouse']
#   tokens: ['The', 'dog', '##house']
#   grams:  [(0, 1), (1, 3)]
# gram 就是 word，不要被名字迷惑
_Gram = collections.namedtuple('_Gram', ['begin', 'end'])


# 每次生成一个 size 大小的数组
def _window(iterable, size):
    i = iter(iterable)
    window = []
    try:
        # 第一次生成
        for e in range(0, size):
            window.append(next(i))
        yield window
    except StopIteration:
        return
    # 后面几次生成
    for e in i:
        # 把 window 往右滑动一次
        # 然后加入一个新的元素
        window = window[1:] + [e]
        yield window


# 判断 grams 是否连续
# 直接判断是否首尾相连就行了
def _contiguous(sorted_grams):
    for a, b in _window(sorted_grams, 2):
        if a.end != b.begin:
            return False
    return True


def _masking_ngrams(grams, max_ngram_size, max_masked_tokens, rng):
    """
    :param grams: 所有的 _Gram，或者称为 word
    :param max_ngram_size: 最多 n 个 gram 连在一起 (word)
    :param max_masked_tokens: 最多有几个 token 被 mask 掉 (word piece)
    :param rng:
    :return: 可以被 mask 的 gram
    """
    if not grams:
        return None

    # 获取 tokens 中的总词数
    grams = sorted(grams)
    num_tokens = grams[-1].end

    for a, b in _window(grams, 2):
        if a.end > b.begin:
            raise ValueError("overlapping grams: {}".format(grams))

    # 因为这个函数要生成 {1, ..., n}-grams
    # 而 n 为 max_ngram_size，因此要循环到 max_ngram_size + 1
    ngrams = {i: [] for i in range(1, max_ngram_size + 1)}
    for gram_size in range(1, max_ngram_size + 1):
        for g in _window(grams, gram_size):
            if _contiguous(g):
                # 将 _window 返回的 grams 合并成一个 _Gram
                ngrams[gram_size].append(_Gram(g[0].begin, g[-1].end))

    # 随机打乱所有 n-grams
    for v in ngrams.values():
        rng.shuffle(v)

    cummulative_weights = list(
        itertools.accumulate([1. / n for n in range(1, max_ngram_size + 1)])
    )

    output_ngrams = []
    masked_tokens = [False] * num_tokens  # mask 的会变成 True，最多 mask 掉 max_masked_tokens 个
    while sum(masked_tokens) < max_masked_tokens and sum(len(s) for s in ngrams.values()):
        # 优先选择包含的 gram 比较多的
        sz = random.choices(range(1, max_ngram_size + 1), cum_weights=cummulative_weights)[0]
        # 如果当前的 n-gram 不符合 mask 的要求
        # 则直接将整个列表 clear 掉
        if sum(masked_tokens) + sz > max_masked_tokens:
            ngrams[sz].clear()
            continue

        if not ngrams[sz]:
            continue

        # 如果当前 n-gram 符合规则
        # 则弹出最后一个（因为前面 shuffle 过）
        gram = ngrams[sz].pop()
        num_gram_tokens = gram.end - gram.begin

        # 由于一个 gram 中可能包含多个 token
        # 因此要再进行一次检查
        if num_gram_tokens + sum(masked_tokens) > max_masked_tokens:
            continue

        # 如果当前 gram 已经被 mask
        if sum(masked_tokens[gram.begin: gram.end]):
            continue

        # 找到了合适的 gram
        masked_tokens[gram.begin: gram.end] = [True] * (gram.end - gram.begin)
        output_ngrams.append(gram)

    return output_ngrams


def _wordpieces_to_grams(tokens):
    # 此函数的作用是识别出 WordPiece tokens 中的所有 word(gram)
    # 不包含 [CLS] 和 [SEP]
    # 原理就是含有 ## 的词的后面一个词一定是一个新词
    # 如果遇到新词，就记录下来上一个词，并将新词的开始位置记录下来
    grams = []
    gram_start_pos = None
    for i, token in enumerate(tokens):
        if gram_start_pos is not None and token.startswith('##'):
            continue
        if gram_start_pos is not None:
            grams.append(_Gram(gram_start_pos, i))
        if token not in ['[CLS]', '[SEP]']:
            gram_start_pos = i
        else:
            gram_start_pos = None

    # 将最后一个词添加进来
    if gram_start_pos is not None:
        grams.append(_Gram(gram_start_pos, len(tokens)))
    return grams


def create_masked_lm_predictions(
        tokens,
        masked_lm_prob,
        max_predictions_per_seq,
        vocab_words,
        rng,
        do_whole_word_mask,
        max_ngram_size=None
):
    # 如果是全词覆盖
    # 需要将 WordPiece 组成词
    if do_whole_word_mask:
        grams = _wordpieces_to_grams(tokens)

    # 否则每一个 token 就是一个词
    # 不包含 [CLS] 和 [SEP]
    else:
        if max_ngram_size:
            raise ValueError('cannot use ngram masking without whole word masking')
        grams = [_Gram(i, i + 1) for i in range(0, len(tokens)) if tokens[i] not in ['[CLS]', '[SEP]']]

    # 计算该样本被 masked 的 token 的数量，注意这里是 token，不是 gram
    # max_predictions_per_seq 是死的，masked_lm_prob * len(tokens) 是活的
    # 所以配合计算出最终结果
    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_grams = _masking_ngrams(
        grams,
        max_ngram_size or 1,
        num_to_predict,
        rng
    )

    masked_lms = []
    output_tokens = list(tokens)
    for gram in masked_grams:
        # 有 80% 的概率替换为 [MASK]
        if rng.random() < 0.8:
            replacement_action = lambda idx: '[MASK]'
        else:
            # 有 10% 的概率不变
            if rng.random() < 0.5:
                replacement_action = lambda idx: tokens[idx]

            # 有 10% 的概率替换成 vocab 中的其他词
            else:
                replacement_action = lambda idx: rng.choice(vocab_words)

        # 每单个位置被 mask 掉
        # 就生成一个 MaskedLmInstance
        # 注意最终输入到 encoder 中的 tokens 是被 mask 过的
        for idx in range(gram.begin, gram.end):
            output_tokens[idx] = replacement_action(idx)
            masked_lms.append(MaskedLmInstance(index=idx, label=tokens[idx]))

        assert len(masked_lms) <= num_to_predict
        masked_lms = sorted(masked_lms, key=lambda x: x.index)

        masked_lm_positions = []
        masked_lm_labels = []
        for p in masked_lms:
            masked_lm_positions.append(p.index)
            masked_lm_labels.append(p.label)

        return output_tokens, masked_lm_positions, masked_lm_labels


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        # 谁长就截断谁
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # 增加一点随机性
        # 掐头去尾
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def main(_):
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))

    logging.info("*** Reading from input files ***")
    for input_file in input_files:
        logging.info("  %s", input_file)

    rng = random.Random(FLAGS.random_seed)
    instances = create_training_instances(
        input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
        FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
        rng, FLAGS.do_whole_word_mask, FLAGS.max_ngram_size
    )

    output_files = FLAGS.output_file.split(",")
    logging.info("*** Writing to output files ***")
    for output_file in output_files:
        logging.info("  %s", output_file)

    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq, output_files,
                                    FLAGS.gzip_compress)


if __name__ == '__main__':
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    app.run(main)
