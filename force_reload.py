#!/usr/bin/env python3
"""
Force reload script to clear Python module cache and restart Flask app
"""
import sys
import os
import shutil
import subprocess

def clear_cache():
    """Clear all Python cache files and directories"""
    print("🧹 Clearing Python cache files...")
    
    # Remove __pycache__ directories
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            cache_dir = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(cache_dir)
                print(f"   ✅ Removed: {cache_dir}")
            except Exception as e:
                print(f"   ❌ Failed to remove {cache_dir}: {e}")
    
    # Remove .pyc files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyc'):
                pyc_file = os.path.join(root, file)
                try:
                    os.remove(pyc_file)
                    print(f"   ✅ Removed: {pyc_file}")
                except Exception as e:
                    print(f"   ❌ Failed to remove {pyc_file}: {e}")

def clear_module_cache():
    """Clear Python's internal module cache"""
    print("🔄 Clearing Python module cache...")
    
    # Clear sys.modules for our custom modules
    modules_to_clear = []
    for module_name in list(sys.modules.keys()):
        if any(name in module_name for name in ['url_multi_labels_predictor', 'url_features_extractor', 'routes', 'config']):
            modules_to_clear.append(module_name)
    
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]
            print(f"   ✅ Cleared module: {module_name}")

def restart_flask():
    """Restart Flask application"""
    print("🚀 Starting Flask application...")
    
    # Set environment variable to prevent bytecode generation
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    
    try:
        # Start Flask app
        subprocess.run([sys.executable, 'run.py'], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Flask application stopped by user")
    except Exception as e:
        print(f"❌ Error starting Flask app: {e}")

if __name__ == "__main__":
    print("🔧 Force Reload Script - Clearing cache and restarting Flask app")
    print("=" * 60)
    
    clear_cache()
    clear_module_cache()
    restart_flask()
