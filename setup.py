from setuptools import setup

setup(
   name="madtree",
   version="1.0",
   description="Generate and solve market configuration trees",
   author="Luigi Foscari",
   author_email="luigi.foscari@unimi.it",
   packages=["madtree"],
   install_requires=["networkx", "pygraphviz", "matplotlib"],
)
