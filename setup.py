from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    """
    This function reads the requirements from a file and returns them as a list.
    """
    requirements = []
    with open(file_path) as file_obj:
        for line in file_obj:
            requirement = line.strip()
            if requirement and not requirement.startswith("#"):  # Ignore empty lines and comments
                requirements.append(requirement)
    return requirements
setup(
name = 'ML_project',
version = '0.0.1',
author= 'Karntapong',
author_email='bladesephiroth01@gmail.com',
packages=find_packages(),
install_requires =get_requirements('requirements.txt')
)