import sys

def error_message_detail(error,error_deatil:sys):
    _,_,exc_tb = error_deatil.exc_info()
    file_name =exc_tb.tb_frame.f_code.co_filename
    error_message = 'Error occured in python scirpt name [{0}] line [{1}] error message[{2}]'.format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    return error_message
class CustomerException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super.__init__(error_message)
        self.error_message = error_message_detail(error_message,error_deatil=error_detail)
    def __str__(self):
        return self.error_message