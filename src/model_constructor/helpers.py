import importlib
from collections import OrderedDict
from functools import partial
from typing import Any, Iterable, Optional, Union

from pydantic import BaseModel
from torch import nn

ListStrMod = list[tuple[str, nn.Module]]
ModSeq = Union[nn.Module, nn.Sequential]


def nn_seq(list_of_tuples: Iterable[tuple[str, nn.Module]]) -> nn.Sequential:
    """return nn.Sequential from OrderedDict from list of tuples"""
    return nn.Sequential(OrderedDict(list_of_tuples))


def init_cnn(module: nn.Module) -> None:
    "Init module - kaiming_normal for Conv2d and 0 for biases."
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, 0)  # type: ignore
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)
    for layer in module.children():
        init_cnn(layer)


def is_module(val: Any) -> bool:
    """Check if val is a nn.Module or partial of nn.Module."""

    to_check = val
    if isinstance(val, partial):
        to_check = val.func
    try:
        return issubclass(to_check, nn.Module)
    except TypeError:
        return False


def instantiate_module(
    name: str,
    default_path: Optional[str] = None,
) -> nn.Module:
    """Instantiate model from name."""
    if default_path is None:
        path_list = name.rsplit(".", 1)
        if len(path_list) == 1:
            default_path = "torch.nn"
            name = path_list[0]
        else:
            if path_list[0] == "nn":
                default_path = "torch.nn"
                name = path_list[1]
            else:
                default_path = path_list[0]
                name = path_list[1]
    try:
        mod = importlib.import_module(default_path)
    except ImportError:
        raise ImportError(f"Module {default_path} not found")
    if hasattr(mod, name):
        module = getattr(mod, name)
        if is_module(module):
            return module
        else:
            raise ImportError(f"Module {name} is not a nn.Module")
    else:
        raise ImportError(f"Module {name} not found at {default_path}")


class Cfg(BaseModel):
    """Base class for config."""

    name: Optional[str] = None

    def _get_str_value(self, field: str) -> str:
        value = getattr(self, field)
        if isinstance(value, type):
            value = value.__name__
        elif isinstance(value, partial):
            value = f"{value.func.__name__} {value.keywords}"
        elif callable(value):
            value = value.__name__
        return value

    def __repr__(self) -> str:
        return f"{self.__repr_name__()}(\n  {self.__repr_str__(chr(10) + '  ')})"

    def __repr_args__(self) -> list[tuple[str, str]]:
        return [
            (field, str_value)
            for field in self.model_fields
            if (str_value := self._get_str_value(field))
        ]

    def __repr_changed_args__(self) -> list[str]:
        """Return list repr for changed fields"""
        return [
            f"{field}: {self._get_str_value(field)}"
            for field in self.model_fields_set  # pylint: disable=E1133
            if field != "name"
        ]

    def print_cfg(self) -> None:
        """Print full config"""
        print(f"{self.__repr_name__()}(\n  {self.__repr_str__(chr(10) + '  ')})")

    def print_changed(self) -> None:
        """Print changed fields."""
        changed_fields = self.__repr_changed_args__()
        if changed_fields:
            print("Changed fields:")
            for i in changed_fields:
                print("  ", i)
        else:
            print("Nothing changed")
