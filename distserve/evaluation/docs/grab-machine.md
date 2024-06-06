# How to Use the Machine Grabbing Script

In this file we will explain how to use the machine grabbing script to grab a `8xA100 SXM 80GB` machine for the OPT-13B and OPT-66B experiments.

First, copy-paste [this script](https://github.com/LLMServe/DistServe/blob/camera-ready/distserve/evaluation/grab-pod.py) to your local machine.

Then, execute the following command:

```bash
python3 grab-pod.py --api-key "API_KEY" --num-gpus 8
```
The `API_KEY` is provided via HotCRP.

When you successfully grab a machine, you will see the following message:

```
Pod deployed successfully
```

After seeing this message, you can visit [RunPod](https://www.runpod.io/), login into the account we provide, switch the account to `Hao Lab@UCSD`, click "Pods" on the left panel, click on the instance with the name "DistServe-AE-GPU", and connect to it via the `SSH over exposed TCP: (Supports SCP & SFTP)` command suggested on the website (it's recommended to remove `-i ~/.ssh/id_ed25519` at the end of the command).

The time to grab the machine ranges from several hours to even days. It is appreciated that you check the script every hour to avoid wasting GPU hours when you get the machine and stop the instance when you finish the review process to save money for us :)

Lastly, if you find it really annoying to grab the machine, you can watch the [screencast](https://drive.google.com/drive/folders/1QCEkpV4Wi2WUutFnDR46NrsSTDXr8lL3?usp=sharing) of experiments reproducing.
