from setuptools import setup, find_packages

setup(
    name="tree_games",
    version="0.1.0",
    description="Recursive LLM Tree Search for Chess",
    author="Claude",
    author_email="user@example.com",
    packages=find_packages(),
    install_requires=[
        "chess",
        "pandas",
        "matplotlib",
        "networkx"
    ],
    extras_require={
        "visualization": ["graphviz"],
        "api": ["openai>=0.27.0"],
        "jupyter": ["ipython", "jupyter", "ipywidgets"]
    },
    entry_points={
        "console_scripts": [
            "chess-tree=tree_games.chess_cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
)