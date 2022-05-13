from setuptools import setup


VERSION_FILENAME = 'model_constructor/version.py'
REQUIREMENTS_FILENAME = 'requirements.txt'
REQUIREMENTS_TEST_FILENAME = 'requirements_test.txt'


# Requirements
try:
    with open(REQUIREMENTS_FILENAME, encoding="utf-8") as fh:
        REQUIRED = fh.read().split("\n")
except FileNotFoundError:
    REQUIRED = []

try:
    with open(REQUIREMENTS_TEST_FILENAME, encoding="utf-8") as fh:
        TEST_REQUIRED = fh.read().split("\n")
except FileNotFoundError:
    TEST_REQUIRED = []

# What packages are optional?
EXTRAS = {"test": TEST_REQUIRED}

# Load the package's __version__ from version.py
version = {}
with open(VERSION_FILENAME, 'r', encoding="utf-8") as fh:
    exec(fh.read(), version)
VERSION = version['__version__']


setup(
    version=VERSION,
    install_requires=REQUIRED,
    extras_require=EXTRAS,
)
