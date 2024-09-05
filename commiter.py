import subprocess

import nbformat
from nbconvert import PythonExporter


def convert_ipynb_to_py(ipynb_path):
    # Load the notebook
    with open(ipynb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Create a Python exporter that includes only code cells
    exporter = PythonExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    exporter.exclude_output = True

    # Convert the notebook to Python code
    (body, _) = exporter.from_notebook_node(nb)

    # Write the Python code to a new file with the same name but .py extension
    py_path = ipynb_path.rsplit(".", 1)[0] + ".py"
    with open(py_path, "w", encoding="utf-8") as f:
        f.write(body)


def git_commit_push(commit_message):
    try:
        # Add all files to staging
        subprocess.run(["git", "add", "-A"], check=True)
        # Commit the changes
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        # Push the changes to the remote repository
        subprocess.run(["git", "push"], check=True)
        print("Changes committed and pushed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


# Example usage
convert_ipynb_to_py("main.ipynb")

commit_message = "default msg"
git_commit_push(commit_message)
# Then use there git pull
