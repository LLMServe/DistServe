import dataclasses

@dataclasses.dataclass
class SimulatorConfig:
    """
    SimulatorConfig: Configuration for the simulator
    """
    
    is_simulator_mode: bool
    profiler_data_path: str   # Path to the profiler data (should be a .json file)
    gpu_mem_size_gb: float       # The size of the GPU memory (in GB)
    