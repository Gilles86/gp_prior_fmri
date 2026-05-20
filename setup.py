from setuptools import setup, find_packages

setup(
    name='gp_prior_fmri',
    version='0.1.0',
    description='Cross-dataset GP-prior pRF fitting and decoding.',
    packages=find_packages(),
    python_requires='>=3.9',   # neural_priors_gp cluster env is on 3.9
    install_requires=[
        'numpy', 'pandas', 'scipy', 'scikit-learn',
        'matplotlib', 'seaborn',
        'nibabel', 'nilearn',
        'pyyaml',
        # braincoder: installed editable from ~/git/braincoder via the env YML
    ],
)
