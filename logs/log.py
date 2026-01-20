# Python script that defines the functions required for logging
# Author: Sasidharan Mahalingam
# Date Created: Jan 19 2026

# Import the required packages
import sys

class TeeFile:
    """
    Class that send the output to both the terminal and the file for logging
    """
    def __init__(self, path):
        """
        Constructor for the TeeFile class

        Arguments:
        ----------
        path - Path to the log file

        Returns:
        --------
        Nothing
        """
        self.file = open(path, "w", buffering=1)
        self.terminal = sys.stdout

    def write(self, s):
        """
        Function that defines the write action

        Arguments:
        ----------
        s - String to write to the terminal and the file

        Returns:
        --------
        Nothing
        """
        # Write to terminal
        self.terminal.write(s)
        self.terminal.flush()
        # Write to file (overwrite)
        self.file.seek(0)
        self.file.truncate()
        self.file.write(s)
        self.file.flush()

    def flush(self):
        """
        Function that defines the flush action
        
        Arguments:
        ----------
        Nothing

        Returns:
        --------
        Nothing
        """
        self.terminal.flush()
        self.file.flush()


class TeeFileAutoFlush:
    """
    Class that auto-flushes file logger and also writes to the terminal
    """
    def __init__(self, path):
        """
        Constructor for the TeTeeFileAutoFlusheFile class

        Arguments:
        ----------
        path - Path to the log file

        Returns:
        --------
        Nothing
        """
        self.file = open(path, "a", buffering=1)
        self.terminal = sys.stdout

    def write(self, s):
        """
        Function that defines the write action

        Arguments:
        ----------
        s - String to write to the terminal and the file

        Returns:
        --------
        Nothing
        """
        # Write to terminal
        self.terminal.write(s)
        self.terminal.flush()
        # Write to file
        self.file.write(s)
        self.file.flush()  # ensure immediate write

    def flush(self):
        """
        Function that defines the flush action
        
        Arguments:
        ----------
        Nothing

        Returns:
        --------
        Nothing
        """
        # flush both
        self.terminal.flush()
        self.file.flush()