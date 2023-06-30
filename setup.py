import re

from setuptools import find_packages, setup


NAME = "gfxmorph"
SUMMARY = "Morphing meshes"

with open(f"{NAME}/__init__.py", "rb") as fh:
    init_text = fh.read().decode()
    VERSION = re.search(r"__version__ = \"(.*?)\"", init_text).group(1)


runtime_deps = [
    "numpy",
    "pygfx",
]

extras_require = {
    "dev": [
        "black",
        "flake8",
        "flake8-black",
        "pep8-naming",
        "pytest",
    ],
}

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(
        exclude=["tests", "tests.*", "examples", "examples.*", "exp", "exp.*"]
    ),
    # package_data={f"{NAME}.data_files": resources_globs},
    python_requires=">=3.8.0",
    install_requires=runtime_deps,
    extras_require=extras_require,
    license="BSD 2-Clause",
    description=SUMMARY,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Almar Klein",
    author_email="almar.klein@gmail.com",
    url="https://github.com/pygfx/gfxmorph",
    data_files=[("", ["LICENSE"])],
    zip_safe=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    # entry_points={
    #     "pyinstaller40": [
    #         "hook-dirs = pygfx.__pyinstaller:get_hook_dirs",
    #         "tests = pygfx.__pyinstaller:get_test_dirs",
    #     ],
    # },
)
