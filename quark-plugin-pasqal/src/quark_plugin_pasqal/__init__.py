"""QUARK plugin for solving QUBOs via Pasqal's neutral-atom QAA (Pulser)."""

from quark_plugin_pasqal.solver import PasqalQAASolver

__all__ = ["PasqalQAASolver", "register"]


def register():
    """Register the Pasqal QAA solver with the QUARK factory.

    This function is called automatically when the plugin is imported via the
    plugin loader. It registers PasqalQAASolver under the name "pasqal_qaa_solver".
    """
    from quark.plugin_manager.factory import register as factory_register

    factory_register("pasqal_qaa_solver", PasqalQAASolver)
