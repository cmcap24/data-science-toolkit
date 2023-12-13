#!/bin/bash

# Stop on first error
set -e

# Read command-line argument for directory path
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 directory_path"
  exit 1
fi

dir_path=$1

# Run mypy for type checking
echo "Running mypy..."
mypy "$dir_path"

# Run isort to sort imports
echo "Running isort..."
isort "$dir_path"

# Run black to format code
echo "Running black..."
black "$dir_path"

# Run flake8 for linting
echo "Running flake8..."
flake8 "$dir_path"

echo "All checks passed!"