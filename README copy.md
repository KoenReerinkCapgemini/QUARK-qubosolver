# PLUGIN_NAME

A valid quark plugin must provide at least one module.
Each module must be a subclass of quark's abstract `Core` class, and must implement the necessary `preprocess` and `postprocess` methods.

This template provides a starting point for creating a QUARK plugin in the recommended structure.
It exposes a single module `ExampleModule` compatible with `QUARK-framework`, missing the necessary implementation of the `preprocess` and `postprocess` functions.
The first steps after creating a new plugin from this template are fixing all TODOs, renaming `ExampleModule` to something more descriptive, and adapting the `pyproject.toml` file to your needs.

This template includes a GitHub action that prepares the newly created plugin for use with quark by exchanging some placeholders like `PLUGIN_NAME`, based on the name you have given your plugin.
This process runs automatically and only takes a few seconds, but you should wait for the action to complete before using the plugin, and pull the changes afterwards.
