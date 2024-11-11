import sys
import subprocess
import importlib.util
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import QApplication

# Custom imports
import global_vars
from gui.pyqt6_gui import VideoPlayer

# List of required packages with version specifications
required_packages = {
    'PyQt6': '6.7.1',
    'opencv-python': '4.10.0.84',
    'numpy': '2.1.1',
    'ultralytics': '8.2.103'
}
python_version = "3.11"  # Python version for running code


def check_python_version(version):
    """Check if a specific version of Python is installed."""
    try:
        # Run the command to check the Python version
        result = subprocess.run(
            [sys.executable, '--version'],
            capture_output=True,
            text=True,
            check=True
        )
        installed_version = result.stdout.strip().split()[1]

        # Split both required and installed versions and compare them
        if installed_version.startswith(version):
            print(f"Python {version} is installed.")
            print(f"Version info: {installed_version}")
        else:
            print(f"Expected Python {version}, but found {installed_version}.")
            print(f"Please install Python {version} from the official Python website.")
            sys.exit(1)  # Exit if the wrong Python version is detected.

    except subprocess.CalledProcessError as e:
        print(f"Error checking Python version: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Python is not installed.")
        print(f"Please install Python {version} from the official Python website.")
        sys.exit(1)


def install_package(package_name, version):
    """Install a package using pip with a specified version."""
    package_with_version = f"{package_name}=={version}"
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_with_version])
    except subprocess.CalledProcessError as e:
        print(f"Error installing package {package_with_version}: {e}")
        sys.exit(1)


def is_package_installed(package_name):
    """Check if a package is installed."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def validate_packages():
    """Ensure all required packages are installed."""
    for package_name, version in required_packages.items():
        if is_package_installed(package_name):
            print(f"{package_name} is already installed.")
        else:
            print(f"{package_name} is not installed. Installing version {version}...")
            install_package(package_name, version)


def main():
    # Validate packages before starting the application
    # validate_packages()

    # PyQt6 Application setup
    QCoreApplication.setApplicationName("Deep Learning Model Analyzer")
    app = QApplication(sys.argv)
    window = VideoPlayer()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    # Check Python version before anything else
    # check_python_version(python_version)
    main()
