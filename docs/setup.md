# Setup

# Nvidia Drivers

Make sure you have `libnvidia-encode-*` and `libnvidia-decode-*` installed.

```
# Replace 535 with your desired version
sudo apt install nvidia-driver-535 libnvidia-encode-535 libnvidia-decode-535
```

If you have an existing driver installation and encounter issues with the above you can try removing the existing installation first:

```
sudo apt purge nvidia-* libnvidia-*
sudo apt autoremove
```
