//go:build !darwin
// +build !darwin

package rnxa

// DetectDevices for non-Darwin systems (CPU only)
func DetectDevices() []Device {
	cpuDevice := Device{
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
