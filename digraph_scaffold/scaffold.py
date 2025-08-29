import os

def scaffold_project(base_dir="fraud_detection_project"):
    """
    Creates the folder and file structure for a fraud detection ML project.
    """

    # Define folders and files
    structure = {
        "data": ["raw/.gitkeep", "processed/.gitkeep"],
        "notebooks": ["eda.ipynb", "model_dev.ipynb"],
        "src": [
            "data_preprocessing.py",
            "feature_engineering.py",
            "supervised_models.py",
            "unsupervised_models.py",
            "deep_learning_models.py",
            "hybrid_models.py",
            "evaluation.py",
            "drift_detection.py",
            "__init__.py"
        ],
        "experiments": ["baseline_experiment.ipynb"],
        "reports": ["report.pdf"],
        "configs": ["config.yaml"],
        "tests": ["test_data_preprocessing.py", "test_models.py"],
    }

    # Create base folder
    os.makedirs(base_dir, exist_ok=True)

    # Iterate and create subfolders + files
    for folder, files in structure.items():
        folder_path = os.path.join(base_dir, folder)
        os.makedirs(folder_path, exist_ok=True)

        for file in files:
            file_path = os.path.join(folder_path, file)
            # Create directories if file has nested folders (like raw/.gitkeep)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Create empty file if it doesn't exist
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    f.write("")

    # Create top-level files
    top_level_files = ["requirements.txt", "README.md", ".gitignore"]
    for file in top_level_files:
        file_path = os.path.join(base_dir, file)
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write("")

    print(f"✅ Project scaffold created at: {base_dir}")


# Run it
if __name__ == "__main__":
    scaffold_project()
