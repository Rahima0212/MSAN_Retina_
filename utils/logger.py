import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import traceback

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, resume=False): 
        """
        Initializes the Logger object.

        Args:
            fpath (str): The path to the log file.
            resume (bool): If True, appends to an existing log file. 
                           If False, overwrites the file.
        """
        self.file = None
        self.resume = resume
        
        # Determine the file open mode based on the resume flag
        mode = 'a' if resume else 'w'
        
        self.file = open(fpath, mode)

    def append(self, target_str):
        """
        Writes a string to both the console and the log file.
        Attempts to convert non-string inputs to strings.

        Args:
            target_str: The object or string to be logged.
        """
        # Ensure the input is a string
        if not isinstance(target_str, str):
            try:
                target_str = str(target_str)
            except Exception:
                # If conversion fails, print the error and exit the function
                traceback.print_exc()
                return

        # Print to console
        print(target_str)
        
        # Write to file and flush the buffer to ensure it's saved immediately
        self.file.write(target_str + '\n')
        self.file.flush()

    def close(self):
        """Closes the log file if it is open."""
        if self.file is not None:
            self.file.close()