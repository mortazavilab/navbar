# navbar/navbar.py
import subprocess
import sys
import os

def run_app():
    # Find the position of '--' separator if it exists
    separator_index = None
    try:
        separator_index = sys.argv.index('--')
    except ValueError:
        pass
    
    # Split arguments at the separator
    if separator_index is not None:
        streamlit_args = sys.argv[1:separator_index]
        app_args = sys.argv[separator_index + 1:]
    else:
        streamlit_args = sys.argv[1:]
        app_args = []
    
    # Get the app.py path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "app.py")
    
    # Verify app.py exists
    if not os.path.exists(app_path):
        print(f"Error: app.py not found at: {app_path}")
        return
    
    # Build streamlit command
    cmd = [sys.executable, "-m", "streamlit", "run"]
    
    # Add any streamlit configuration arguments
    cmd.extend(streamlit_args)
    
    # Add app.py
    cmd.append(app_path)
    
    # Add separator and app arguments
    if app_args:
        cmd.append('--')
        cmd.extend(app_args)
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStreamlit app stopped.")
    except Exception as e:
        print(f"Error running Streamlit app: {e}")

if __name__ == "__main__":
    run_app()
