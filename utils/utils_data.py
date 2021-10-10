import logging
import os
import torch

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for classification."""

    def __init__(self, guid, sentence, label):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The label for the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.sentence = sentence
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, guid, input_ids, input_mask, segment_ids, label_ids):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            splits = line.split('\t', 1)
            if len(splits) > 1:
                label = splits[0]
                sentence = splits[1]
            else:
                label = -1
                sentence = splits[0]
            examples.append(InputExample(guid="{}-{}".format(mode, guid_index), sentence=sentence, label=label))
            guid_index += 1

    return examples


def convert_examples_to_features_spare(
        examples,
        label_list,
        max_seq_length,
        tokenizer
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d", ex_index, len(examples))

        inputs = tokenizer.encode_plus(example.sentence, padding='max_length', truncation=True, max_length=max_seq_length, return_tensors="pt")
        input_ids = inputs['input_ids'][0].numpy()
        input_mask = inputs['attention_mask'][0].numpy()
        segment_ids = inputs['token_type_ids'][0].numpy()

        if example.label == -1:
            label_ids = None
        else:
            label_ids = [label_map[example.label]]

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=label_ids
                          )
        )
    return features

def convert_examples_to_features(
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        window = 50
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        using sliding window at token level to deal with long text
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d", ex_index, len(examples))

        # valid_mask = []

        guid = example.guid
        tokens = tokenizer.tokenize(example.sentence)

        if example.label == -1:
            label_ids = None
        else:
            label_ids = [label_map[example.label]]

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        # sliding window at token level (deal with long text)
        if len(tokens) > max_seq_length - special_tokens_count:
            start = 0
            span = max_seq_length - special_tokens_count
            while start + span < len(tokens):
                tokens_now = tokens[start:start+span]
                input_ids, input_mask, segment_ids = padding_mask(tokens_now, max_seq_length, tokenizer, cls_token_at_end,
                                                                  cls_token,
                                                                  cls_token_segment_id, sep_token, sep_token_extra,
                                                                  pad_on_left, pad_token,
                                                                  pad_token_segment_id, sequence_a_segment_id,
                                                                  mask_padding_with_zero)
                features.append(
                    InputFeatures(guid=guid,
                                  input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_ids=label_ids)
                )
                start += window
        else:
            input_ids, input_mask, segment_ids = padding_mask(tokens, max_seq_length, tokenizer, cls_token_at_end,
                                                              cls_token,
                                                              cls_token_segment_id, sep_token, sep_token_extra,
                                                              pad_on_left, pad_token,
                                                              pad_token_segment_id, sequence_a_segment_id,
                                                              mask_padding_with_zero)
            features.append(
                InputFeatures(guid=guid,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids)
            )

    return features

def padding_mask(
        tokens,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,):

    special_tokens_count = 3 if sep_token_extra else 2
    assert len(tokens) <= max_seq_length - special_tokens_count

    tokens += [sep_token]

    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if cls_token_at_end:
        tokens += [cls_token]
        segment_ids += [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
    else:
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    # if ex_index < 3:
        # logger.info("*** Example ***")
        # logger.info("guid: %s", example.guid)
        # logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        # logger.info("valid_mask: %s", " ".join([str(x) for x in valid_mask]))
        # logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        # logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        # logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        # logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
        # logger.info("start_ids: %s", " ".join([str(x) for x in start_ids]))
        # logger.info("end_ids: %s", " ".join([str(x) for x in end_ids]))

    return input_ids, input_mask, segment_ids


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    batch_tuple = tuple(map(torch.stack, zip(*batch)))
    batch_lens = torch.sum(batch_tuple[1], dim=-1, keepdim=False)
    max_len = batch_lens.max().item()
    results = ()
    for item in batch_tuple:
        if item.dim() >= 2:
            results += (item[:, :max_len],)
        else:
            results += (item,)
    return results


def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        return labels
