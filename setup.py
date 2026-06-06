from setuptools import setup, find_packages

setup(
    name='mmpsurvnet',
    version='1.0.0',
    description='MultiModalPreopSurvNet: preoperative multimodal glioblastoma '
                'survival prediction on UCSF-PDGM.',
    author='',
    packages=find_packages(exclude=('scripts', 'tests')),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.0',
        'numpy',
        'pandas',
        'scipy',
        'scikit-image',
        'nibabel',
        'SimpleITK>=2.4.0',
        'scikit-learn',
        'scikit-survival',
        'lifelines',
        'matplotlib',
        'tqdm',
    ],
)
