# curl --request POST \
#   --header 'content-type: application/json' \
#   --url 'https://api.runpod.io/graphql?api_key=${YOUR_API_KEY}' \
#   --data 
  
# {"input":{"cloudType":"SECURE","containerDiskInGb":20,"volumeInGb":0,"deployCost":1.48,"gpuCount":2,"gpuTypeId":"NVIDIA GeForce RTX 4090","minMemoryInGb":62,"minVcpuCount":16,"startJupyter":true,"startSsh":true,"templateId":"runpod-torch-v21","volumeKey":null,"ports":"8888/http,22/tcp","dataCenterId":"US-OR-1","networkVolumeId":"dg0br51h50","name":"spider-test"}}

# {"operationName":"Mutation","variables":{"input":{"cloudType":"SECURE","containerDiskInGb":1024,"volumeInGb":0,"deployCost":4.58,"gpuCount":2,"gpuTypeId":"NVIDIA A100-SXM4-80GB","minMemoryInGb":250,"minVcpuCount":32,"startJupyter":true,"startSsh":true,"templateId":"xbivg6n3b6","volumeKey":null,"ports":"8080/http,22/tcp,8000/tcp","dataCenterId":"US-OR-1","networkVolumeId":"dg0br51h50","name":"spider-test"}},"query":"mutation Mutation($input: PodFindAndDeployOnDemandInput) {\n  podFindAndDeployOnDemand(input: $input) {\n    id\n    imageName\n    env\n    machineId\n    machine {\n      podHostId\n      __typename\n    }\n    __typename\n  }\n}"}
  
import os, sys
import argparse
import requests
import time
import random

def request_pod(api_key: str, num_gpus: int):
    data = '{"query": "mutation { podFindAndDeployOnDemand( input: { cloudType: SECURE, gpuCount: ' + str(num_gpus) + ', volumeInGb: 64, containerDiskInGb: 512, minVcpuCount: 32, minMemoryInGb: 128, gpuTypeId: \\"NVIDIA A100-SXM4-80GB\\", name: \\"DistServe-AE-GPU\\", startJupyter: false, startSsh: true, templateId: \\"xbivg6n3b6\\", volumeKey: null, dockerArgs: \\"\\", ports: \\"8080/http,22/tcp,8000/tcp\\", dataCenterId: null, volumeMountPath: \\"/workspace\\", networkVolumeId:null, env: [] } ) { id imageName env machineId machine { podHostId } } }"}'
    print("Payload:", data)
    response = requests.post('https://api.runpod.io/graphql?api_key=' + api_key, headers={'content-type': 'application/json'}, data=data)
    print(response.text)
    json_data = response.json()
    if "errors" in json_data:
        error = json_data["errors"][0]
        msg = error["message"]
        if msg != "There are no longer any instances available with the requested specifications. Please refresh and try again.":
            print("Alert! Unknown error: ", error)
            sys.exit(1)
    else:
        if "data" not in json_data:
            print("Error:", json_data)
            return
        print("Pod deployed successfully")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, required=True, help="API key")
    parser.add_argument("--num-gpus", type=int, default=8)
    args = parser.parse_args()
    
    while True:
        try:
            request_pod(args.api_key, args.num_gpus)
        except Exception as e:
            print("Error:", e)
            time.sleep(10)
        time.sleep(1 + random.randint(-5, 5)/10)
    
