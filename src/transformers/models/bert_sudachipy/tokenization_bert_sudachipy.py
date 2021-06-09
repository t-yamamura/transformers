from typing import Any, Callable, List, Optional


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
                >>> tokens = []
                >>> for m in x:
                >>>     if m.part_of_speech()[0] in conjugation:
                >>>>         tokens.append(m.surface())
                >>>     else:
                >>>         tokens.append(m.normalized_form())
                >>> return tokens
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

        sudachi_dict = dictionary.Dictionary(config_path=config_path, resource_dir=resource_dir, dict_type=dict_type)
        self.sudachi = sudachi_dict.create()

    def tokenize(self, text, **kwargs):
        return self.formatter(self.sudachi.tokenize(text, self.split_mode))
