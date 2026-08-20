"""QUARK plugin for LP to QUBO conversion using qiskit-optimization."""

from quark_plugin_lp_to_qubo.lp_to_qubo import LpToQuboConverter

__all__ = ["LpToQuboConverter", "register"]


def register():
    """Register the LP to QUBO converter with the QUARK factory.
    
    This function is called automatically when the plugin is imported via the
    plugin loader. It registers LpToQuboConverter under the name "lp_to_qubo_converter".
    """
    from quark.plugin_manager.factory import register as factory_register

    factory_register("lp_to_qubo_converter", LpToQuboConverter)
