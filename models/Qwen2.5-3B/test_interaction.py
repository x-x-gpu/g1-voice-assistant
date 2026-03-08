import subprocess
import time
import threading
import sys
import os

def read_stream(stream, prefix, output_list):
    """Reads from a stream and prints with a prefix."""
    for line in iter(stream.readline, ''):
        print(f"{prefix}{line}", end='')
        output_list.append(line)
    stream.close()

def run_test():
    # Use -u for unbuffered output
    cmd = [sys.executable, '-u', 'run.py']
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Starting process: {cmd} in {cwd}")
    
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
        cwd=cwd
    )

    stdout_list = []
    stderr_list = []
    
    t_out = threading.Thread(target=read_stream, args=(process.stdout, "", stdout_list))
    t_err = threading.Thread(target=read_stream, args=(process.stderr, "[ERR] ", stderr_list))
    
    t_out.start()
    t_err.start()
    
    inputs = [
        "你好",
        "你是谁",
        "向前走", # Implicit
        "以0.5m/s的速度向前移动", # Explicit
        "向左移动",
        "原地旋转",
        "停止",
        "挥手",
        "握手",
        "比个心",
        "双手举高",
        "exit"
    ]

    try:
        # Wait for "小智已就绪" message roughly
        print("Waiting for initialization...")
        time.sleep(10) 
        
        for i, inp in enumerate(inputs):
            print(f"\n--- [Test {i+1}/{len(inputs)}] Sending: {inp} ---")
            if process.poll() is not None:
                print("Process ended unexpectedly.")
                break
                
            try:
                process.stdin.write(inp + "\n")
                process.stdin.flush()
            except OSError as e:
                print(f"Error writing to stdin: {e}")
                break
                
            # Wait for response. Tools might take longer.
            # We don't have a reliable way to know when it's done without parsing stdout in real-time complexly.
            # A fixed sleep is simpler for this test.
            time.sleep(5) 

    except Exception as e:
        print(f"Test exception: {e}")
    finally:
        if process.poll() is None:
            print("\nTerminating process...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        
        t_out.join(timeout=1)
        t_err.join(timeout=1)
        
        # Save full log
        log_content = "".join(stdout_list)
        err_content = "".join(stderr_list)
        
        with open('test_interaction_log.txt', 'w', encoding='utf-8') as f:
            f.write("=== STDOUT ===\n")
            f.write(log_content)
            f.write("\n=== STDERR ===\n")
            f.write(err_content)
            
        print(f"Log saved to test_interaction_log.txt")

if __name__ == "__main__":
    run_test()
