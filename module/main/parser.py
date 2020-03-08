"""
THIS SPECIFIC FILE IS DISTRIBUTED UNDER THE UNLICENSE: http://unlicense.org.

THIS MEANS YOU CAN USE THIS CODE EXAMPLE TO KICKSTART A PROJECT YOUR OWN.
AFTER YOU CREATED YOUR OWN ORIGINAL WORK, YOU CAN REPLACE THIS HEADER :)
"""

import sys
import argparse

from .myclass import RunMe


def main(name=None):
    """This function is called when run as python3 -m ${MODULE}
    Parse any additional arguments and call required module functions."""

    if sys.argv:
        # called through CLI
        module_name = __loader__.name.split('.')[0]
        parser = argparse.ArgumentParser(
            prog=module_name,
            description="This is my new shiny pip package called: " + module_name,
        )

        parser.add_argument('--name', action='store', nargs=1, required=False, type=str,
                            default=['dummy'],
                            help="Add a name")

        args = parser.parse_args(sys.argv[1:])

        if args.name:
            # argparser provides us a list, even if only one argument
            if isinstance(args.name, list) and isinstance(args.name[0], str):
                name = args.name[0]


    instance = RunMe(name=name)
    sys.stdout.write(instance.say_hello() + "\n")
    return 0
