# How to Use the Machine Grabbing Script

In this file we will explain how to use the machine grabbing script to grab a machine for the OPT-13B and OPT-66B experiments.

First, copy-paste (this file)[https://github.com/LLMServe/DistServe/blob/camera-ready/distserve/evaluation/grab-pod.py] to your local machine, and prepare a SSH public key.

Then, execute the following command:

```bash
python3 grab-pod.py --api-key YOUR_API_KEY --public-key "YOUR_SSH_PUBKEY" --num-gpus 8
```

We will provide you with an API key via HotCRP.

For example:

```bash
python3 grab-pod.py --api-key THIS_IS_AN_API_KEY --public-key "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFoz16d3eU3q48NWvR2JXGcaMmuHOHsE/g8gVSzsJixh intlsy@i" --num-gpus 8
```
