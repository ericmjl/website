"""Lektor plugin package metadata for marimo islands."""

from setuptools import setup

setup(
    name="lektor-marimo-islands",
    version="0.1.0",
    description="Inline marimo island fragments into Lektor pages.",
    py_modules=["lektor_marimo_islands"],
    entry_points={
        "lektor.plugins": [
            "marimo-islands = lektor_marimo_islands:MarimoIslandsPlugin",
        ]
    },
)
