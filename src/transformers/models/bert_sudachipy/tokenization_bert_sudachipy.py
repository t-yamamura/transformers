import collections
import copy
import os
from typing import Any, Callable, List, Optional, Tuple

from ..bert_japanese.tokenization_bert_japanese import CharacterTokenizer
from ..bert.tokenization_bert import load_vocab, WordpieceTokenizer
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


class BertSudachipyTokenizer(PreTrainedTokenizer):
    """"""

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            do_subword_tokenize=True,
            subword_tokenizer_type="wordpiece",
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            sudachipy_kwargs=None,
            **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            do_lower_case=do_lower_case,
            do_subword_tokenize=do_subword_tokenize,
            subword_tokenizer_type=subword_tokenizer_type,
            sudachipy_kwargs=sudachipy_kwargs,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocabulary file at path '{vocab_file}'.")

        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.sudachipy_kwargs = copy.deepcopy(sudachipy_kwargs)

        self.word_tokenizer = SudachipyTokenizer(**(self.sudachipy_kwargs or {}))

        self.do_subword_tokenize = do_subword_tokenize
        self.subword_tokenizer_type = subword_tokenizer_type
        if do_subword_tokenize:
            if subword_tokenizer_type == "wordpiece":
                self.subword_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)
            elif subword_tokenizer_type == "character":
                self.subword_tokenizer = CharacterTokenizer(vocab=self.vocab, unk_token=self.unk_token)
            else:
                raise ValueError(f"Invalid subword_tokenizer_type '{subword_tokenizer_type}' is specified.")

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def __getstate__(self):
        state = dict(self.__dict__)
        del state["word_tokenizer"]
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.word_tokenizer = SudachipyTokenizer(**(self.sudachipy_kwargs or {}))

    def _tokenize(self, text, **kwargs):
        tokens = self.word_tokenizer.tokenize(text)

        if self.do_subword_tokenize:
            split_tokens = [sub_token for token in tokens for sub_token in self.subword_tokenizer.tokenize(token)]
        else:
            split_tokens = tokens

        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)


class SudachipyTokenizer:
    """Runs tokenization with SudachiPy."""

    def __init__(
            self,
            split_mode: Optional[str] = "C",
            config_path: Optional[str] = None,
            resource_dir: Optional[str] = None,
            dict_type: Optional[str] = "core",
            word_form: Optional[str] = "surface",
            formatter: Optional[Callable[[Any], List[str]]] = None
    ):
        """
        Constructs a SudachipyTokenizer.

        Args:
            split_mode (:obj:`str`, `optional`, defaults to :obj:`"C"`):
                The mode of splitting.
                "A", "B", or "C" can be specified.
            config_path (:obj:`str`, `optional`, defaults to :obj:`None`):
                Path to a config file of SudachiPy to be used for the sudachi dictionary initialization.
            resource_dir (:obj:`str`, `optional`, defaults to :obj:`None`):
                Path to a resource dir containing resource files, such as "sudachi.json".
            dict_type (:obj:`str`, `optional`, defaults to :obj:`"core"`):
                Sudachi dictionary type to be used for tokenization.
                "small", "core", or "full" can be specified.
            word_form (:obj:`str`, `optional`, defaults to :obj:`"surface"`):
                Word form of a morpheme after tokenization.
                "surface", "dictionary", or "normalized" can be specified.
            formatter (:obj:`Callable`, `optional`, defaults to :obj:`None`):
                Custom tokenization formatter.

                >>> def custom_formatter(x):
                >>>     # Use normalized_form() except for conjugated_words
                >>>     conjugation = ['動詞', '形容詞', '形容動詞', '助動詞']
                >>>     tokens = []
                >>>     for m in x:
                >>>         if m.part_of_speech()[0] in conjugation:
                >>>             tokens.append(m.surface())
                >>>         else:
                >>>             tokens.append(m.normalized_form())
                >>>     return tokens
        """
        self.config_path = config_path
        self.resource_dir = resource_dir
        self.dict_type = dict_type.lower()
        self.word_form = word_form.lower()

        try:
            from sudachipy import tokenizer
            from sudachipy import dictionary
        except ModuleNotFoundError as error:
            raise error.__class__(
                "You need to install SudachiPy to use SudachipyTokenizer. "
                "See https://github.com/WorksApplications/SudachiPy for installation. "
            )

        split_mode = split_mode.lower()
        if split_mode == "c":
            self.split_mode = tokenizer.Tokenizer.SplitMode.C
        elif split_mode == "b":
            self.split_mode = tokenizer.Tokenizer.SplitMode.B
        elif split_mode == "a":
            self.split_mode = tokenizer.Tokenizer.SplitMode.A
        else:
            raise ValueError("Invalid `split_mode` is specified.")

        if formatter:
            self.formatter = formatter
        else:
            if word_form == 'surface':
                self.formatter = lambda x: [m.surface() for m in x]
            elif word_form == 'dictionary':
                self.formatter = lambda x: [m.dictionary_form() for m in x]
            elif word_form == 'normalized':
                self.formatter = lambda x: [m.normalized_form() for m in x]
            else:
                raise ValueError("Invalid `word_form` is specified.")

        self.sudachi_dict = dictionary.Dictionary(config_path=config_path, resource_dir=resource_dir, dict_type=dict_type)
        self.sudachi = self.sudachi_dict.create()

    def tokenize(self, text, **kwargs):
        return self.formatter(self.sudachi.tokenize(text, self.split_mode))
