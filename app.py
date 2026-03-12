import sys
import os

# Add project1-eval-toolkit to path so its relative imports (llm_judge, text_compare) resolve
project1_dir = os.path.join(os.path.dirname(__file__), "project1-eval-toolkit")
sys.path.insert(0, project1_dir)

exec(open(os.path.join(project1_dir, "app.py")).read())
