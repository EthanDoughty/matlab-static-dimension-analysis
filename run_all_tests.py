# Ethan Doughty
# run_all_tests.py
import os
from matlab_parser import parse_matlab
from analysis import analyze_program

TEST_FILES = [f"tests/test{i}.m" for i in range(1, 11)]


def run_test(path: str):
    print(f"===== Analysis for {path}")
    if not os.path.exists(path):
        print(f"ERROR: file not found\n")
        return

    with open(path, "r") as f:
        src = f.read()

    try:
        ast = parse_matlab(src)
    except Exception as e:
        print(f"Error while parsing {path}: {e}\n")
        return

    env, warnings = analyze_program(ast)

    if not warnings:
        print("No dimension warnings.")
    else:
        print("Warnings:")
        for w in warnings:
            print("-", w)

    print("Final environment:")
    print("   ", env)
    print()


def main():
    for path in TEST_FILES:
        run_test(path)


if __name__ == "__main__":
    main()