VOCAB_BIAS = ['']
NUM_LABELS = 2
LABELS_INT_TO_STRING_DICT = {0: 'neutral', 1: 'hate'}
LABELS_STRING_TO_INT_DICT = {'neutral': 0, 'hate': 1}


def labels_int_to_string(lbl):
    return LABELS_INT_TO_STRING_DICT[lbl]


def labels_string_to_int(lbl):
    return LABELS_STRING_TO_INT_DICT[lbl]
