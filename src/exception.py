import sys 
from src.logger import logging

def error_message_detail(error, error_detail:sys):
    _,_,exc_tb= error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno,str(error)

    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message

# Essai pour voir si ça fonctionne

if __name__=="__main__" :
   try:
       a=1/0
   except Exception as e:
        logging.info("Divide by zero")
        raise CustomException(e, sys)
    

# Aprés exécution du fichier l'erreur suivante ---> :     
# __main__.CustomException: Error occured in python script name [C:\Users\Lenovo\Documents\DSPython\P7_Credit_Scoring\src\exception.py] line number [24] error message[division by zero]
    