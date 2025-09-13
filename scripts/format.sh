#!/bin/bash

# Code formatting script
# Automatically format all code using black and isort

echo "🎨 Formatting code..."

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must be run from project root directory"
    exit 1
fi

# Format with black
echo "📝 Running black formatter..."
uv run black .

# Sort imports with isort
echo "📝 Running isort..."
uv run isort .

echo "✅ Code formatting complete!"