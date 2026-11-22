This practical project focuses on the automated detection of skin cancer through clinical image analysis. Utilizing Machine Learning and Convolutional Neural Networks (CNNs), the system classifies skin lesions based on the methodologies reviewed in the study Automatic Skin Cancer Detection Using Clinical Images: A Comprehensive Review (https://pmc.ncbi.nlm.nih.gov/articles/PMC10672549/)

The datasets used to conduct this project were collected from ISIC (International Skin images Collaborators) specifically the "Challenge 2017: Test", "Challenge 2017: Training" and "Challenge 2017: Validation".

This dataset is composed by Dermoscopy images, these are high-magnification digital photographs of skin lesions, captured with a specialized medical instrument called a dermatoscope.

The link to access is https://api.isic-archive.com/collections/?pinned=true


## 3D IMAGE VISUALIZATION TOOL

For better visualization approach I used "NAPARI" python Library [https://napari.org/stable/getting_started/installation.html#install-python-package]

To install napari you can follow the steps bellow.

While not strictly required, it is highly recommended to install napari into a clean virtual environment using the uv venv environment manager.

This should be set up before you install napari. For example, setting up a Python 3.14 environment with uv venv:

```bash
uv venv -p 3.14
source .venv/bin/activate # for macOS and Linux
.venv\Scripts\activate # for Windows
```
napari can be installed from PyPI on most macOS, Linux, and Windows systems with Python 3.11-3.14 using uv pip:

```bash
uv pip install "napari[all]"
```

You can then upgrade napari to a new version using:

```bash
uv pip install "napari[all]" --upgrade
```
(See Choosing a different Qt backend below for an explanation of the [all] notation.)

(See Using constraints file for help installing older versions of napari)