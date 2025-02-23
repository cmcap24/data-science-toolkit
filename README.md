# Data Science Toolkit

## Overview
The **Data Science Toolkit** is a comprehensive Python package designed to support various facets of data science, including:
- **Causal Inference**: Methods for estimating causal effects.
- **Machine Learning**: Training, evaluation, and model interpretation.
- **Statistical Analysis**: Hypothesis testing and probability distributions.
- **Data Processing**: Feature engineering and preprocessing.
- **Visualization**: Advanced data visualization techniques.

This repository is built with **Python 3.13** and managed using **Poetry** for dependency management.

---

## Installation
### **Prerequisites**
Ensure you have the following installed:
- [Python 3.13](https://www.python.org/)
- [Poetry](https://python-poetry.org/docs/)
- [Git](https://git-scm.com/)

### **Clone the Repository**
```bash
git clone https://github.com/cmcap24/data-science-toolkit.git
cd data-science-toolkit
```

### **Set Up the Environment**
```bash
poetry install
```

If you're using `pyenv`, make sure to use Python 3.13:
```bash
pyenv local 3.13.0
```

---

## Usage
### **Activating the Virtual Environment**
```bash
poetry shell
```

### **Running Pre-Commit Hooks**
```bash
pre-commit run --all-files
```

### **Example Usage**
You can start using the toolkit in your Python scripts:
```python
# TODO: add example usage
```

---

## Project Structure
```
data-science-toolkit/
│── src/                     # Source code
│   ├── causal_inference/     # Causal inference models
│   ├── machine_learning/     # ML models
│   ├── data_processing/      # Data transformations
│   ├── visualization/        # Visualization utilities
│   ├── statistics/           # Statistical methods
│
│── tests/                    # Unit tests
│── notebooks/                # Jupyter notebooks for experiments
│── data/                     # Datasets (ignored in Git)
│── pyproject.toml            # Poetry dependencies
│── .pre-commit-config.yaml   # Pre-commit hooks
│── README.md                 # Project documentation
│── .gitignore                # Ignored files
```

---

## Style Guide
To maintain code consistency and readability, follow these guidelines:
- **Code Formatting**: Use [Black](https://black.readthedocs.io/) for automatic code formatting.
- **Linting**: Use [Flake8](https://flake8.pycqa.org/) to enforce coding standards.
- **Import Sorting**: Use [isort](https://pycqa.github.io/isort/) to organize imports.
- **Typing**: Use type hints (`mypy`) to ensure type safety.
- **Naming Conventions**:
  - Use `snake_case` for variables and functions.
  - Use `PascalCase` for class names.
  - Use `UPPER_CASE` for constants.
- **Docstrings**: Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for writing docstrings.

To automatically enforce these rules, run:
```bash
pre-commit run --all-files
```

---

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Add new feature"`).
4. Push to your fork (`git push origin feature-branch`).
5. Open a Pull Request.

---

## License
This project is licensed under the **MIT License**.

---

## Contact
For questions or contributions, reach out via GitHub Issues or email **chris.capozzola@gmail.com**.
