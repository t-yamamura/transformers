from typing import TYPE_CHECKING

from ...file_utils import _BaseLazyModule


_import_structure = {
    "tokenization_bert_sudachipy": ["BertSudachipyTokenizer", "SudachipyTokenizer"],
}


if TYPE_CHECKING:
    from .tokenization_bert_sudachipy import BertSudachipyTokenizer, SudachipyTokenizer

else:
    import importlib
    import os
    import sys

    class _LazyModule(_BaseLazyModule):
        """
        Module class that surfaces all objects but only performs associated imports when the objects are requested.
        """

        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        def _get_module(self, module_name: str):
            return importlib.import_module("." + module_name, self.__name__)

    sys.modules[__name__] = _LazyModule(__name__, _import_structure)
