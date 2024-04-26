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

When you successfully grab a machine, you will see the following message:

```
Pod deployed successfully
```

After seeing this message, you can visit [RunPod](https://www.runpod.io/), login into the account we provide, click "Pods" on the left panel, click on the instance with the name "DistLLM-AE-GPU", and connect to it via the "SSH over exposed TCP: (Supports SCP & SFTP)" command suggested on the website (it's recommended to remove `-i ~/.ssh/id_ed25519` at the end of the command). (@zym: polish this)

(@zym 委婉地提醒一下 reviewer 最好一直盯着脚本，在机器开起来之后就用，省点钱)

Lastly, if you find it really annoying to grab the machine, you can watch our [screencast](https://drive.google.com/drive/folders/1QCEkpV4Wi2WUutFnDR46NrsSTDXr8lL3?usp=sharing).
