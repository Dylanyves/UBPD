from .single import SingleTrainer
from .cv import CVTrainer


def get_trainer(cv, *args, **kwargs):
    if cv:
        return CVTrainer(*args, **kwargs)
    else:
        return SingleTrainer(*args, **kwargs)
