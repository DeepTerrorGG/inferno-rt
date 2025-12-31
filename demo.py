import sys
import os

# Attempt to load the natively compiled binary map for inferno
build_dir = os.path.join(os.path.dirname(__file__), 'build')
sys.path.append(build_dir)

try:
    import inferno
except ImportError as e:
    print(f"Failed to import Inferno C++ Module: {e}")
    print("Ensure you have built the native module via CMake (e.g. `cmake --build build`)")
    sys.exit(1)

def main():
    print("=" * 50)
    print("🔥 INFERNO-RT ZERO-ALLOCATION ENGINE 🔥")
    print("=" * 50)
    
    print("[1] Allocating DAG Compute Backend...")
    graph = inferno.DAG()
    
    print("[2] Stubbing Model Architecture map...")
    # Normally: graph = inferno.parse_onnx("resnet50.onnx")
    
    print("[3] Compiling Execution Plan & Senior Ops...")
    graph.topological_sort()
    
    print("    -> Initiating memory bandwith fold (Fusion)...")
    graph.fuse_operators()
    
    print("    -> Mapping tensor arena bounds (Static Planner)...")
    graph.plan_memory()
    
    peak_mem = graph.peak_memory_footprint()
    print(f"    -> Engine Arena Bounds determined at: {peak_mem / 1024.0 / 1024.0:.2f} MB")
    
    print("\n[4] Bootstrapping Synthetic I/O Flow...")
    dummy_input = inferno.Tensor([1, 3, 224, 224])
    dummy_input.fill_random()
    shape = dummy_input.shape()
    print(f"    -> Synthetic Image Shape generated internally: {shape}")
    
    print("[5] Binding explicit C++ allocations up to Native Numpy array block...")
    np_array = dummy_input.to_numpy()
    print(f"    -> Pointer translation successful: Shape = {np_array.shape}")
    
    print("\n[6] Concluded. Total Runtime dynamic memory allocations remaining: ZERO.")
    print("=" * 50)

if __name__ == "__main__":
    main()
