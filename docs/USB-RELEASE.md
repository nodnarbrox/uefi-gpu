# UEFI GPU USB image — experimental bring-up release

This download is the archived RTX 3060 GPU bring-up USB artifact, repackaged
with accurate instructions and notices. It is separate from the later 177B
inference checkpoint published in the repository. It contains no model weights
and does not provide a ready-to-use LLM chat service.

Contents: x86-64 UEFI application at EFI/BOOT/BOOTX64.EFI, a 128 MiB FAT ESP,
and a raw 65 MiB partition carrying signed, unmodified NVIDIA 570.144 GSP
firmware. Total image size: 210 MiB. Use on compatible NVIDIA hardware only.
The original EFI and GSP payload are unchanged. README/license files were added
to the ESP for publication; the packaged image has its own SHA256.

## Use

1. Download the .img and SHA256SUMS. In PowerShell, use
   `Get-FileHash .\uefi-gpu-usb-bringup-v0.1.0.img -Algorithm SHA256`.
2. Write the image to a spare USB drive using an image writer's raw/DD mode.
   Writing the image erases the selected USB drive; verify the device first.
3. Choose UEFI boot. This research EFI is unsigned, so Secure Boot must permit
   it. Boot from the USB using the one-time boot menu.
4. Capture the complete output. GPU discovery and booter status are diagnostic
   results, not proof of a completed inference run. The app may write
   GPURESULT.TXT on the boot volume.

Recorded inference results elsewhere use Linux/KVM/VFIO and OVMF with an RTX
3060 passed through. This release has not been freshly boot-tested on a physical
machine. Compatibility with other boards/GPUs is not established. The old
bring-up binary is not claimed to match the later driver source checkpoint.

The NVIDIA firmware remains NVIDIA's software under NVIDIA-DRIVER-LICENSE.txt.
Its redistribution terms concern software supplied for use with open-source
operating systems; the documented research host is Linux/KVM. No license to
firmware is granted by this project. The firmware was not patched or signed
by the project. Additional upstream notices are included in the ESP.

Source and evidence: https://github.com/nodnarbrox/uefi-gpu
