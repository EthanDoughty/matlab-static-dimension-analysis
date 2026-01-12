# Ethan Doughty
# ast_printer.py
import sys
import pprint

from matlab_parser import parse_matlab


def parse_file(path: str):
    """Read a Mini-MATLAB file and return its AST"""
    with open(path, "r") as f:
        src = f.read()
    return parse_matlab(src)


def main():
    if len(sys.argv) != 2:
        print("Usage: python ast_printer.py <file.m>")
        sys.exit(1)

    path = sys.argv[1]
    try:
        ast = parse_file(path)
    except Exception as e:
        print(f"Error while parsing {path}: {e}")
        sys.exit(1)

    print(f"==== AST for {path}")
    pprint.pprint(ast, width=80, compact=False)


if __name__ == "__main__":
    main()