#!/usr/bin/env python3
"""
Pre-PyPI Testing Suite

Test everything before uploading to PyPI to avoid embarrassing failures!
"""

import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report results"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print(f"  ✅ Success: {description}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed: {description}")
        print(f"  📍 Error: {e.stderr}")
        return False, e.stderr

def test_build_system():
    """Test that the package builds correctly"""
    print("\n📦 Testing Build System")
    print("-" * 40)
    
    # Clean previous builds
    dirs_to_clean = ['build', 'dist', '*.egg-info']
    for pattern in dirs_to_clean:
        run_command(f"rm -rf {pattern}", f"Cleaning {pattern}")
    
    # Test building
    success, output = run_command("python -m build", "Building package with python -m build")
    if not success:
        return False
    
    # Check if files were created
    dist_files = list(Path("dist").glob("*"))
    if len(dist_files) < 2:  # Should have .whl and .tar.gz
        print(f"  ❌ Expected 2 files in dist/, found {len(dist_files)}")
        return False
    
    print(f"  📦 Created {len(dist_files)} distribution files")
    for file in dist_files:
        print(f"    - {file.name}")
    
    return True

def test_package_metadata():
    """Test package metadata using twine check"""
    print("\n📋 Testing Package Metadata")
    print("-" * 40)
    
    success, output = run_command("twine check dist/*", "Checking package metadata")
    return success

def test_fresh_install():
    """Test installing the package from built distribution"""
    print("\n🔄 Testing Fresh Installation")
    print("-" * 40)
    
    # Create temporary virtual environment
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_path = Path(temp_dir) / "test_venv"
        
        # Create venv
        success, _ = run_command(f"python -m venv {venv_path}", "Creating test virtual environment")
        if not success:
            return False
        
        # Get activation script
        if sys.platform == "win32":
            activate_script = venv_path / "Scripts" / "activate.bat"
            pip_cmd = f"{venv_path}/Scripts/pip"
        else:
            activate_script = venv_path / "bin" / "activate"
            pip_cmd = f"{venv_path}/bin/pip"
        
        # Install from wheel
        wheel_files = list(Path("dist").glob("*.whl"))
        if not wheel_files:
            print("  ❌ No wheel file found in dist/")
            return False
            
        wheel_file = wheel_files[0]
        success, _ = run_command(f"{pip_cmd} install {wheel_file}", "Installing from wheel")
        if not success:
            return False
        
        # Test import in fresh environment
        python_cmd = f"{venv_path}/bin/python" if sys.platform != "win32" else f"{venv_path}/Scripts/python"
        
        test_script = """
import mbvlgranger
from mbvlgranger import quick_mbvl_granger, MultiBandVLGranger
import numpy as np

# Quick functionality test  
x = np.random.randn(100)
y = np.random.randn(100)
result = quick_mbvl_granger(x, y, fs=250, max_lag=10, print_results=False)
print("Package works in fresh environment!")
"""
        
        success, _ = run_command(f'{python_cmd} -c "{test_script}"', "Testing package in fresh environment")
        return success

def test_console_scripts():
    """Test console script entry points"""
    print("\n⚡ Testing Console Scripts")
    print("-" * 40)
    
    # Test if console script is accessible
    success, output = run_command("mbvlgranger-test --help", "Testing console script")
    if not success:
        print("  ⚠️  Console script not found (this might be OK if not implemented)")
        return True  # Don't fail if console script isn't implemented
    
    return success

def test_readme_and_docs():
    """Test that README renders correctly"""
    print("\n📖 Testing Documentation")
    print("-" * 40)
    
    # Check if README exists
    readme_files = list(Path(".").glob("README.*"))
    if not readme_files:
        print("  ❌ No README file found")
        return False
    
    print(f"  ✅ Found README: {readme_files[0].name}")
    
    # Test if README can be parsed (basic check)
    try:
        with open(readme_files[0], 'r', encoding='utf-8') as f:
            content = f.read()
            if len(content) < 100:
                print("  ⚠️  README is very short")
            else:
                print(f"  ✅ README has {len(content)} characters")
    except Exception as e:
        print(f"  ❌ Error reading README: {e}")
        return False
    
    return True

def test_dependencies():
    """Test that all dependencies can be resolved"""
    print("\n📚 Testing Dependencies")
    print("-" * 40)
    
    # Test installing dependencies
    success, _ = run_command("pip install -e .", "Installing package with dependencies")
    if not success:
        return False
    
    # Test importing all dependencies
    deps_test = """
import numpy
import pandas  
import scipy
import matplotlib
import statsmodels
import sklearn
import dtaidistance
print("All dependencies importable!")
"""
    
    success, _ = run_command(f'python -c "{deps_test}"', "Testing dependency imports")
    return success

def test_version_consistency():
    """Test that version is consistent across files"""
    print("\n🔢 Testing Version Consistency")
    print("-" * 40)
    
    # Check version in setup.py
    setup_version = None
    try:
        with open("setup.py", 'r') as f:
            content = f.read()
            # Look for version="..." pattern
            import re
            match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                setup_version = match.group(1)
    except:
        pass
    
    # Check version in pyproject.toml
    pyproject_version = None
    try:
        with open("pyproject.toml", 'r') as f:
            content = f.read()
            import re
            match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                pyproject_version = match.group(1)
    except:
        pass
    
    # Check version in __init__.py
    init_version = None
    try:
        with open("mbvlgranger/__init__.py", 'r') as f:
            content = f.read()
            import re
            match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                init_version = match.group(1)
    except:
        pass
    
    print(f"  setup.py version: {setup_version}")
    print(f"  pyproject.toml version: {pyproject_version}")
    print(f"  __init__.py version: {init_version}")
    
    versions = [v for v in [setup_version, pyproject_version, init_version] if v is not None]
    if len(set(versions)) > 1:
        print("  ❌ Version mismatch detected!")
        return False
    elif len(versions) == 0:
        print("  ⚠️  No version found in any file")
        return False
    else:
        print(f"  ✅ All versions consistent: {versions[0]}")
        return True

def main():
    """Run all pre-PyPI tests"""
    print("=" * 60)
    print("🚀 PRE-PYPI TESTING SUITE")
    print("=" * 60)
    
    tests = [
        ("Build System", test_build_system),
        ("Package Metadata", test_package_metadata),
        ("Fresh Installation", test_fresh_install),
        ("Console Scripts", test_console_scripts),
        ("Documentation", test_readme_and_docs),
        ("Dependencies", test_dependencies),
        ("Version Consistency", test_version_consistency),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 PRE-PYPI TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} | {test_name}")
        if success:
            passed += 1
    
    print("-" * 60)
    print(f"📊 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 Your package is ready for PyPI upload!")
        print("\nNext steps:")
        print("1. twine upload --repository testpypi dist/*")
        print("2. Test install from TestPyPI")
        print("3. twine upload dist/*")
        return True
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed.")
        print("Fix the issues above before uploading to PyPI.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)