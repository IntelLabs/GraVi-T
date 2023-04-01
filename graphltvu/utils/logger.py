import os
import logging

def get_logger(path_result, file_name, file_mode='w'):
    """
    Get the logger that logs runtime messages under "path_result"
    """

    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.DEBUG,
                        handlers=[logging.StreamHandler(),
                        logging.FileHandler(filename=os.path.join(path_result, f'{file_name}.log'), mode=file_mode)])

    return logging.getLogger()
