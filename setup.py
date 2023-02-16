
from setuptools import setup, find_packages


requirements = ["Bio", 'embed', 'h5py', 'matplotlib', 'numpy', 'pandas', 'positional_encodings',\
                'scikit_learn', 'scipy', 'seaborn', 'tape', 'torch', 'tqdm', 'transformers']
# TODO - specify loose version requirements? E.g. Bio>1.0.0. Make this even more permissive?
        # Removed dscript & pymol to be more permissive

with open("README.md", "r") as readme_file:
    readme = readme_file.read()


setup(
    name="abml",
    version="0.0.1",
    author="Rohit Singh, Chiho Im, Taylor Sorenson",
    author_email="rohitsingh@gmail.com",
    description="AbMAP: antibody-specific embeddings to apply to any foundational PLM",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/rs239/ablm",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: TODO:: TODO",
    ],
)