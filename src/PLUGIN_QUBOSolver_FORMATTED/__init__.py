from quark.plugin_manager import factory

from src.PLUGIN_QUBOSolver_FORMATTED.QUBOSolver_module import QuboSolverModule


def register() -> None:
    """
    Register all modules exposed to quark by this plugin.
    For each module, add a line of the form:
        factory.register("module_name", Module)

    The "module_name" will later be used to refer to the module in the configuration file.
    """
    factory.register("qubo_solver", QuboSolverModule)
