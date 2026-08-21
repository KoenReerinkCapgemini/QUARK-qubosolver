"""QUARK plugin for solving QUBOs with the Pasqal QUBO solver."""

from quark_plugin_pasqal.qubo_to_pasqal import QuboToPasqal
from quark_plugin_pasqal.simple_qubo_provider import SimpleQuboProvider

__all__ = ["QuboToPasqal", "SimpleQuboProvider", "register"]


def register() -> None:
    """Register the Pasqal solver module with the QUARK factory."""
    from quark.plugin_manager.factory import register as factory_register

    factory_register("qubo_to_pasqal", QuboToPasqal)
    factory_register("simple_qubo_provider", SimpleQuboProvider)
