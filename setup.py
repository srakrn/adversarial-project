import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="clustre",  # Replace with your own username
    version="0.0.1",
    author="Sirakorn Lamyai",
    author_email="sirakorn.l@ku.th",
    description="Clustering package for adversarial paper",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"clustre": "clustre"},
    packages=["clustre"],
    python_requires=">=3.6",
)
