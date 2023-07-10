from functools import partial

from pytest import CaptureFixture
from torch import nn

from model_constructor.helpers import Cfg, instantiate_module, is_module


class Cfg2(Cfg):
    int_value: int = 10


def test_is_module():
    """test is_module"""
    assert not is_module("some string")
    assert is_module(nn.Module)
    assert is_module(nn.ReLU)
    assert not is_module(nn)
    assert is_module(partial(nn.ReLU, inplace=True))
    assert not is_module(partial(int, "10"))


def test_instantiate_module():
    """test instantiate_module"""
    mod = instantiate_module("ReLU")
    assert mod is nn.ReLU
    mod = instantiate_module("nn.Tanh")
    assert mod is nn.Tanh
    mod = instantiate_module("torch.nn.SELU")
    assert mod is nn.SELU
    # wrong name
    try:
        mod = instantiate_module("wrong_name")
    except ImportError as err:
        assert str(err) == "Module wrong_name not found at torch.nn"
    # wrong module
    try:
        mod = instantiate_module("wrong_module.some_name")
    except ImportError as err:
        assert str(err) == "Module wrong_module not found"
    # not nn.Module
    try:
        mod = instantiate_module("model_constructor.helpers.instantiate_module")
    except ImportError as err:
        assert str(err) == "Module instantiate_module is not a nn.Module"


def test_cfg_repr_print(capsys: CaptureFixture[str]):
    """test repr and print results"""
    cfg = Cfg()
    repr_res = cfg.__repr__()
    assert repr_res == "Cfg(\n  )"
    cfg.print_set_fields()
    out = capsys.readouterr().out
    assert out == "Nothing changed\n"
    cfg.name = "cfg_name"
    repr_res = cfg.__repr__()
    assert repr_res == "Cfg(\n  name='cfg_name')"
    cfg.print_cfg()
    out = capsys.readouterr().out
    assert out == "Cfg(\n  name='cfg_name')\n"
    # Set fields. default - name is not in changed
    cfg = Cfg2(name="cfg_name")
    cfg.print_set_fields()
    out = capsys.readouterr().out
    assert out == "Nothing changed\n"
    assert "name" in cfg.model_fields_set
    cfg = Cfg2(int_value=0)
    cfg.print_set_fields()
    out = capsys.readouterr().out
    assert out == "Set fields:\nint_value: 0\n"
    # Changed fields
    cfg = Cfg2(name="cfg_name")
    assert cfg.changed_fields == {"name": "cfg_name"}
    cfg.int_value = 1
    cfg.name = None
    assert cfg.changed_fields == {"int_value": 1}
    # print
    cfg.print_changed_fields()
    out = capsys.readouterr().out
    assert out == "Changed fields:\nint_value: 1\n"
    # return to default
    cfg.int_value = 10
    assert not cfg.changed_fields
    cfg.print_changed_fields()
    out = capsys.readouterr().out
    assert out == "Nothing changed\n"
