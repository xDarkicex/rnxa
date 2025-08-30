//go:build darwin
// +build darwin

package rnxa

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework Foundation

#include "internal/metal/metal_ops.h"
*/
import "C"

// DetectDevices discovers available Metal devices (Darwin only)
func DetectDevices() []Device {
	var devices []Device

	// Try to create Metal device using the same functions as metal engine
	metalDevice := C.metal_create_device()
	if metalDevice != nil {
		defer C.metal_release_device(metalDevice)

		namePtr := C.get_device_name_safe(metalDevice)
		name := C.GoString(namePtr)
		cores := int(C.get_device_cores_safe(metalDevice))
		memory := estimateDeviceMemory(name)

		device := Device{
			ID:       0,
			Name:     name,
			Type:     GPU,
			Memory:   memory,
			Cores:    cores,
			Platform: "Metal",
		}
		devices = append(devices, device)
	}

	// Always include CPU as fallback
	cpuDevice := Device{
		ID:       len(devices),
		Name:     "CPU",
		Type:     CPU,
		Memory:   8 * 1024 * 1024 * 1024,
		Cores:    8,
		Platform: "CPU",
	}
	devices = append(devices, cpuDevice)

	return devices
}

// Helper functions remain the same
func estimateDeviceMemory(deviceName string) uint64 {
	switch {
	case contains(deviceName, "M3 Max"):
		return 36 * 1024 * 1024 * 1024
	case contains(deviceName, "M3 Pro"):
		return 18 * 1024 * 1024 * 1024
	case contains(deviceName, "M3"):
		return 16 * 1024 * 1024 * 1024
	case contains(deviceName, "M2 Max"):
		return 32 * 1024 * 1024 * 1024
	case contains(deviceName, "M2 Pro"):
		return 16 * 1024 * 1024 * 1024
	case contains(deviceName, "M2"):
		return 16 * 1024 * 1024 * 1024
	default:
		return 16 * 1024 * 1024 * 1024
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr ||
		(len(s) > len(substr) && findSubstring(s, substr)))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		match := true
		for j := 0; j < len(substr); j++ {
			if s[i+j] != substr[j] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

func GetBestDevice() Device {
	devices := DetectDevices()
	for _, device := range devices {
		if device.Platform == "Metal" {
			return device
		}
	}
	return devices[len(devices)-1]
}

func IsMetalAvailable() bool {
	devices := DetectDevices()
	for _, device := range devices {
		if device.Platform == "Metal" {
			return true
		}
	}
	return false
}
