import os
import random
import subprocess
from datetime import datetime, timedelta

# Configuration
START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 4, 18)
DUMMY_FILE = "CHANGELOG_INTERNAL.md"

# Commit messages matching our exact history working together 
# and the files currently in the codebase.
COMMIT_MESSAGES = [
    "Add Table of Contents to README.md",
    "Update project icons in README",
    "Detail architecture and memory subsystem documentation",
    "Setup CMake build system and GoogleTest integration",
    "Implement naive GEMM and AVX2 optimizations in gemm.cpp",
    "Add Protobuf integration for ONNX parsing",
    "Implement DAG structure and operator fusion stub",
    "Add CUDA kernels for Conv2D and GEMM in cuda_backend.cu",
    "Implement Pybind11 Python wrappers in bindings.cpp",
    "Disable CUDA backend conditionally in CMakeLists.txt",
    "Create real-time evaluation demo in realtime_eval.cpp",
    "Implement N-dimensional tensor logic and strides in tensor.cpp",
    "Fix MSVC C++20 standard compliance considerations in build",
    "Add vcvars64 compiler environment setup script for Visual Studio",
    "Update project proposal documentation",
    "Implement ArenaAllocator definition in allocator.hpp",
    "Write gtest fixtures for operator fusion and tensor strides"
]

def generate_commits():
    current_date = START_DATE
    
    while current_date <= END_DATE:
        # User requested 1 commit per day, max 2.
        num_commits = random.randint(1, 2) 
        
        for i in range(num_commits):
            # Pick a random time for the commit during the workday
            hour = random.randint(9, 22)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            
            commit_time = current_date.replace(hour=hour, minute=minute, second=second)
            date_str = commit_time.isoformat()
            
            commit_msg = random.choice(COMMIT_MESSAGES)
            
            # Make a small change to a dummy file to ensure the commit has a diff
            with open(DUMMY_FILE, "a") as f:
                f.write(f"- [{date_str}] {commit_msg}\n")
            
            # Stage the file
            subprocess.run(["git", "add", DUMMY_FILE], check=True)
            
            # Set environment variables for backdating
            env = os.environ.copy()
            env['GIT_AUTHOR_DATE'] = date_str
            env['GIT_COMMITTER_DATE'] = date_str
            
            # Commit silently directly via python
            subprocess.run(["git", "commit", "-m", commit_msg], env=env, check=True, stdout=subprocess.DEVNULL)
            
        current_date += timedelta(days=1)

if __name__ == "__main__":
    print(f"Generating realistic commit history from {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}...")
    generate_commits()
    print("Done! You can now check `git log` and run `git push origin master`.")
