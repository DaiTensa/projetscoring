from setuptools import find_packages, setup 
from typing import List


HYPEN_E_DOT ='-e .'
def get_requirment(file_path: str)->List[str]:
    """Cette Fonction retourne la liste de requierments

    Args:
        file_path (str): chemin pour accèder au fichier requierments

    Returns:
        List[str]: la liste des packages ou librairies
    """

    requierments=[]
    with open(file_path) as file_obj:
        requierments= file_obj.readlines()
        requierments = [req.replace("\n"," ") for req in requierments ]

        if HYPEN_E_DOT in requierments:
            requierments.remove(HYPEN_E_DOT)
    return requierments



setup(
name= 'scoring',
version= '0.0.1',
author= 'Dai',
author_email= 'dai.tensaout@gmail.com',
packages= find_packages(),
install_requires= get_requirment('requirements.txt') 

)