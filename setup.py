"""Setup script for the AniMAIRE package."""

from pathlib import Path
from setuptools import find_packages, setup

this_directory = Path(__file__).parent


def read_requirements(requirements_path: Path) -> list[str]:
    if not requirements_path.is_file():
        return []
    return [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


install_requires = read_requirements(this_directory / "requirements.txt")
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name='AniMAIRE',
    packages=find_packages(exclude='pytests'),
    package_data={"AniMAIRE":[
                                "anisotropic_MAIRE_engine/data/*.csv",
                                "anisotropic_MAIRE_engine/rigidityPredictor/data/*.pkl"
                                         ]},
    version='1.4.4',
    description='Python library for running the anisotropic version of MAIRE+',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Space Environment and Protection Group, University of Surrey',
    license='CC By-NC-SA 4.0',
    url='https://github.com/ssc-maire/AniMAIRE-public',
    keywords='anisotropic MAIRE+ atmospheric ionizing radiation dose rates cosmic rays ground level enhancements GLEs protons alpha particles neutrons effective ambient equivalent aircraft aviation Earth solar system sun space magnetic field',
    install_requires=install_requires,
    python_requires=">=3.10",
    setup_requires=['pytest-runner','wheel'],
    tests_require=['pytest'],
)
