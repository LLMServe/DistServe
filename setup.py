from setuptools import setup

setup(
    name="distserve",
    version="0.0.1",
    author="Yinmin Zhong, Shengyu Liu, Junda Chen",
    description="Disaggregated inference engine for LLMs.",
    packages=["distserve", "simdistserve"],
    zip_safe=False,
)
