import logging

from models.bigbirdpegasus import BigBirdPegasus
from models.pegasusx import PegasusX
from models.bart import Bart
from models.longt5 import LongT5
from models.hfmodel import HFModel

translate_model_name = {
    "bigbirdpegasus": BigBirdPegasus,
    "pegasusx": PegasusX,
    "bart": Bart,
    "longt5": LongT5,
}


def load_model(
    model: str,
) -> HFModel:

    assert model in translate_model_name
    return translate_model_name[model]()
