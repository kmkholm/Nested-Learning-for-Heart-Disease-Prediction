# Contributing to Nested Learning for Heart Disease Prediction

First off, thank you for considering contributing to this project! 🎉

This project implements cutting-edge research (NeurIPS 2025) and welcomes contributions from the community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Style Guidelines](#style-guidelines)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Areas for Contribution](#areas-for-contribution)

---

## Code of Conduct

### Our Pledge

We pledge to make participation in this project a harassment-free experience for everyone, regardless of:
- Age, body size, disability, ethnicity
- Gender identity and expression
- Level of experience, education
- Nationality, personal appearance, race, religion
- Sexual identity and orientation

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards others

**Unacceptable behavior includes:**
- Trolling, insulting/derogatory comments, personal attacks
- Public or private harassment
- Publishing others' private information
- Other conduct inappropriate in a professional setting

---

## How Can I Contribute?

### 1. Reporting Bugs

Before submitting a bug report:
- Check existing issues to avoid duplicates
- Collect information about the bug

**Good bug reports include:**
- Clear, descriptive title
- Exact steps to reproduce
- Expected vs actual behavior
- Screenshots if applicable
- Environment details (OS, Python version, PyTorch version)

**Example:**
```markdown
**Bug**: Training fails with CUDA out of memory

**To Reproduce**:
1. Load dataset with batch_size=64
2. Run model.train()
3. Error occurs after epoch 2

**Expected**: Training should complete
**Actual**: CUDA OOM error

**Environment**:
- OS: Ubuntu 20.04
- Python: 3.8.10
- PyTorch: 2.0.1
- CUDA: 11.7
```

### 2. Suggesting Enhancements

Enhancement suggestions are welcome! Include:
- Clear, descriptive title
- Detailed explanation of the enhancement
- Why this enhancement would be useful
- Possible implementation approach

### 3. Code Contributions

See [Development Process](#development-process) below.

---

## Getting Started

### Fork and Clone

```bash
# Fork the repository on GitHub

# Clone your fork
git clone https://github.com/YOUR_USERNAME/nested-learning-heart-disease.git
cd nested-learning-heart-disease

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/nested-learning-heart-disease.git
```

### Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### Run Tests

```bash
# Run the main script to verify installation
python nested_learning_heart_disease.py

# Run tests (if available)
pytest tests/
```

---

## Development Process

### 1. Create a Branch

```bash
# Update your fork
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write clean, readable code
- Add docstrings to functions/classes
- Include type hints
- Add comments for complex logic
- Update documentation if needed

### 3. Test Your Changes

```bash
# Run the main script
python nested_learning_heart_disease.py

# Test specific components
python -c "from nested_learning_heart_disease import DeepMomentumOptimizer; print('OK')"
```

### 4. Format Code

```bash
# Format with black
black nested_learning_heart_disease.py

# Check style
flake8 nested_learning_heart_disease.py

# Type checking
mypy nested_learning_heart_disease.py
```

---

## Style Guidelines

### Python Style Guide

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these specifics:

**Line Length**: 100 characters max (not 80)

**Imports**:
```python
# Standard library
import os
import sys

# Third-party
import numpy as np
import torch
import torch.nn as nn

# Local
from nested_learning_heart_disease import NestedLearningClassifier
```

**Naming Conventions**:
```python
# Classes: PascalCase
class DeepMomentumOptimizer:
    pass

# Functions/methods: snake_case
def train_model():
    pass

# Constants: UPPER_SNAKE_CASE
LEARNING_RATE = 0.001

# Private: prefix with underscore
def _internal_function():
    pass
```

### Documentation Style

**Module Docstrings**:
```python
"""
Brief description of the module.

Detailed description if needed.
"""
```

**Function Docstrings** (Google style):
```python
def function_name(param1: type1, param2: type2) -> return_type:
    """
    Brief description.
    
    Detailed description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is invalid
        
    Example:
        >>> result = function_name(1, 2)
        >>> print(result)
        3
    """
```

**Class Docstrings**:
```python
class ClassName:
    """
    Brief description.
    
    Detailed description.
    
    Attributes:
        attr1: Description of attr1
        attr2: Description of attr2
        
    Example:
        >>> obj = ClassName()
        >>> obj.method()
    """
```

### Type Hints

Always use type hints:
```python
from typing import List, Dict, Tuple, Optional

def process_data(
    data: List[int],
    config: Dict[str, float]
) -> Tuple[np.ndarray, Optional[str]]:
    """Process data with configuration."""
    pass
```

---

## Commit Messages

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **test**: Adding/updating tests
- **chore**: Maintenance tasks

### Examples

**Good commit messages:**
```
feat(optimizer): Add Deep Momentum Optimizer

Implement DeepMomentumOptimizer class with:
- Delta-rule enhancement
- Deep memory compression
- Configurable memory depth

Closes #42

---

fix(training): Fix gradient clipping in trainer

Gradient clipping was not applied correctly.
Now properly clips gradients before optimizer step.

Fixes #38

---

docs(readme): Update installation instructions

Add conda installation option and troubleshooting section.
```

**Bad commit messages:**
```
Update code
Fix bug
WIP
asdf
```

---

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No merge conflicts

### Submitting

1. **Push to your fork**:
```bash
git push origin feature/your-feature-name
```

2. **Create Pull Request** on GitHub:
   - Use descriptive title
   - Reference related issues
   - Describe changes clearly
   - Add screenshots if relevant

3. **PR Template**:
```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Related Issues
Closes #XX

## Testing
Describe how you tested the changes.

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass
```

### Review Process

- Maintainers will review your PR
- Address feedback promptly
- Make requested changes
- Once approved, PR will be merged

---

## Areas for Contribution

### High Priority

1. **New Optimizers**
   - Implement additional deep optimizer variants
   - Add Hessian-based optimizers
   - Explore different preconditioning strategies

2. **Memory Architectures**
   - Design new continuum memory structures
   - Implement learned frequency adaptation
   - Add attention-based memory

3. **Benchmarking**
   - Test on more medical datasets
   - Compare with recent papers
   - Add performance profiling

4. **Documentation**
   - Create Jupyter notebook tutorials
   - Add video walkthroughs
   - Improve API documentation

### Medium Priority

5. **Visualization**
   - Add frequency importance analysis
   - Visualize gradient flow per level
   - Create interactive dashboards

6. **Testing**
   - Add unit tests
   - Add integration tests
   - Add performance tests

7. **Optimization**
   - GPU optimization
   - Mixed precision training
   - Distributed training support

### Nice to Have

8. **Applications**
   - Apply to NLP tasks
   - Apply to computer vision
   - Multi-modal learning

9. **Deployment**
   - Add Docker support
   - Create REST API
   - Add model serving guides

10. **Education**
    - Create tutorial series
    - Add beginner-friendly examples
    - Record video explanations

---

## Questions?

- 📧 **Email**: kmkhol01@gmail.com
- 💬 **Issues**: Use GitHub issues for questions
- 📖 **Docs**: Check existing documentation first

---

## Recognition

Contributors will be:
- Listed in README.md
- Credited in release notes
- Acknowledged in papers/presentations (for significant contributions)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Nested Learning! 🙏

**Together, we're advancing AI research and making it accessible to everyone.** 🚀
