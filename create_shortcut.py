import os
import sys
import platform
import subprocess

def create_shortcut():
    system = platform.system()
    
    if system == "Darwin":  # macOS
        # AppleScriptを使用してショートカットを作成
        desktop_path = os.path.expanduser("~/Desktop")
        script_path = os.path.abspath("run_app.py")
        
        applescript = f'''
        tell application "Finder"
            make new alias file at POSIX file "{desktop_path}" to POSIX file "{script_path}"
            set name of result to "株価分析アプリ"
        end tell
        '''
        
        subprocess.run(["osascript", "-e", applescript])
        print("デスクトップにショートカットを作成しました。")
        
    elif system == "Windows":
        # Windows用のショートカット作成
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        path = os.path.join(desktop, "株価分析アプリ.lnk")
        
        target = os.path.abspath("run_app.py")
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = sys.executable
        shortcut.Arguments = f'"{target}"'
        shortcut.WorkingDirectory = os.path.dirname(target)
        shortcut.save()
        print("デスクトップにショートカットを作成しました。")
        
    else:
        print("このOSはサポートされていません。")

if __name__ == "__main__":
    create_shortcut() 