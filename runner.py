# Ethan Doughty
# runner.py
import sys

from matlab_parser import parse_matlab
from analysis import analyze_program


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 runner.py <file.m>")
        sys.exit(1)

    path = sys.argv[1]
    try:
        with open(path, "r") as f:
            src = f.read()
        ast = parse_matlab(src)
    except Exception as e:
        print(f"Error while parsing {path}: {e}")
        sys.exit(1)

    env, warnings = analyze_program(ast)

    print(f"=== Analysis for {path} ===")
    if not warnings:
        print("No dimension warnings.")
    else:
        print("Warnings:")
        for w in warnings:
            print("  -", w)

    print("\nFinal environment:")
    print(env)


if __name__ == "__main__":
    main()