//go:build !darwin && !linux

package rnxa

// DetectDevices for non-Darwin, non-Linux systems (CPU only).
// Linux gets its own device_linux.go that probes NVIDIA GPUs
// via nvidia-smi. Darwin gets device_darwin.go (MPS + Metal +
// CPU). This file covers Windows for now — Windows CUDA
// detection is a follow-up to the linux-first CUDA cut.
func DetectDevices() []Device {
	cpuDevice := Device{
		// TODO: query actual host RAM + core count via OS-specific syscalls
		// (syscall.Sysctl on darwin, /proc/cpuinfo and /proc/meminfo on linux).
		ID:       0,
		Name:     "CPU",
		Type:     CPU,
		Memory:   8 * 1024 * 1024 * 1024,
		Cores:    8,
		Platform: "CPU",
	}
	return []Device{cpuDevice}
}

func GetBestDevice() Device {
	devices := DetectDevices()
	return devices[0]
}

func IsMetalAvailable() bool {
	return false
}
