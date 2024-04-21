import argparse

from distserve.simulator.utils import ReqResult, Dataset, load_req_result_list

def compare(
    std_reqs: list[ReqResult],
    sim_reqs: list[ReqResult]
):
    num_prompts = len(std_reqs)
    if num_prompts != len(sim_reqs):
        raise ValueError(f"Number of prompts in the standard and simulated results are different: {num_prompts} vs {len(sim_reqs)}")

    for i in range(num_prompts):
        std_req = std_reqs[i]
        sim_req = sim_reqs[i]
        if std_req.prompt_len != sim_req.prompt_len or std_req.output_len != sim_req.output_len:
            print(f"Prompt length or output length mismatch at index {i}: {std_req.prompt_len} vs {sim_req.prompt_len}, {std_req.output_len} vs {sim_req.output_len}")
            print("Falling back to sorting by prompt length and output length")
            std_reqs.sort(key=lambda x: (x.prompt_len, x.output_len))
            sim_reqs.sort(key=lambda x: (x.prompt_len, x.output_len))
            compare(std_reqs, sim_reqs)
            return
        print(f"{std_req.prompt_len:4d} {std_req.output_len:4d} {std_req.ttft_ms:8.2f} {sim_req.ttft_ms:8.2f} ({(sim_req.ttft_ms-std_req.ttft_ms)/std_req.ttft_ms*100:5.1f}%) {std_req.tpot_ms:8.2f} {sim_req.tpot_ms:8.2f} {(sim_req.tpot_ms-std_req.tpot_ms)/std_req.tpot_ms*100:5.1f} %")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--std", type=str, required=True, help="Path to the standard exp result")
    parser.add_argument("--sim", type=str, required=True, help="Path to the simulated exp result")
    args = parser.parse_args()
    
    std_reqs = load_req_result_list(args.std)
    sim_reqs = load_req_result_list(args.sim)
    
    compare(std_reqs, sim_reqs)
    