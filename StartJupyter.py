import os
import sys
import subprocess

def start_jupyter_notebook(project_path = './data'):
    print(f"Starting Jupyter notebook in folder: {project_path}")
    os.chdir(project_path)
    subprocess.run(["jupyter", "notebook", "--port", "2205"])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        project_path = sys.argv[1]
        start_jupyter_notebook(project_path)
    else:
        start_jupyter_notebook()