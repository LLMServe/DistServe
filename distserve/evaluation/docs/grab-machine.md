# How to Use the Machine Grabbing Script

In this file we will explain how to use the machine grabbing script to grab a machine for the OPT-13B and OPT-66B experiments.

First, copy-paste [this script](https://github.com/LLMServe/DistServe/blob/camera-ready/distserve/evaluation/grab-pod.py) to your local machine, and copy the SSH public key you give to us.

Then, execute the following command:

```bash
python3 grab-pod.py --api-key "API_KEY" --public-key "YOUR_SSH_PUBKEY" --num-gpus 8
```

The API key is provided via HotCRP.

For example:

```bash
python3 grab-pod.py --api-key THIS_IS_AN_API_KEY --public-key "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFoz16d3eU3q48NWvR2JXGcaMmuHOHsE/g8gVSzsJixh intlsy@i" --num-gpus 8
```

TODO: explain what will happen when you grab the machine successfully and what to do next. @lsy