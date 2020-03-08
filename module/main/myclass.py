"""
THIS SPECIFIC FILE IS DISTRIBUTED UNDER THE UNLICENSE: http://unlicense.org.

THIS MEANS YOU CAN USE THIS CODE EXAMPLE TO KICKSTART A PROJECT YOUR OWN.
AFTER YOU CREATED YOUR OWN ORIGINAL WORK, YOU CAN REPLACE THIS HEADER :)
"""

import sys

class RunMe:
    """Example class for education and testing."""
    def __init__(self, name=None):
        self.name = ""
        if self.update(name) is None:
            raise TypeError("Error updating name\n")

    def update(self, name):
        """Update name"""
        if not isinstance(name, str):
            sys.stderr.write("That is not a string! Refusing to work.\n")
            return None
        self.name = name
        return self

    def say_hello(self):
        """Return name attribute."""
        return str("Hi " + self.name + "!")

    # add more fun stuff here
