#!/bin/bash

# Code quality check script
# Run all linting and formatting tools

echo "🔧 Running code quality checks..."

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must be run from project root directory"
    exit 1
fi

# Function to run a command and check its exit status
run_check() {
    local cmd="$1"
    local name="$2"

    echo "📝 Running $name..."
    if eval "$cmd"; then
        echo "✅ $name passed"
        return 0
    else
        echo "❌ $name failed"
        return 1
    fi
}

# Initialize failure counter
failures=0

# Run black in check mode
run_check "uv run black --check --diff ." "Black formatting check" || ((failures++))

# Run isort in check mode
run_check "uv run isort --check-only --diff ." "Import sorting check" || ((failures++))

# Run flake8
run_check "uv run flake8 ." "Flake8 linting" || ((failures++))

# Run mypy
run_check "uv run mypy ." "MyPy type checking" || ((failures++))

# Summary
echo ""
if [ $failures -eq 0 ]; then
    echo "🎉 All quality checks passed!"
    exit 0
else
    echo "💥 $failures quality check(s) failed"
    echo ""
    echo "To fix formatting issues, run:"
    echo "  ./scripts/format.sh"
    exit 1
fi