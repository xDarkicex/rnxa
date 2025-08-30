package rnxa

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework Foundation

#include "internal/metal/metal_ops.h"

// Use the same typedefs as metal_ops.h to avoid type conflicts
const char* get_device_name_safe(MTLDeviceRef device);
int get_device_cores_safe(MTLDeviceRef device);
*/
import "C"

// DetectDevices discovers available Metal devices
func DetectDevices() []Device {
	var devices []Device

	// Use the same Metal device creation as metal_ops.m
	metalDevice := C.metal_create_device()
	if metalDevice != nil {
		defer C.metal_release_device(metalDevice)

		// Use our safe wrapper functions
		namePtr := C.get_device_name_safe(metalDevice)
		name := C.GoString(namePtr)
		cores := int(C.get_device_cores_safe(metalDevice))

		// Estimate memory based on device name
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
		Memory:   8 * 1024 * 1024 * 1024, // 8GB default
		Cores:    8,                      // Approximate CPU cores
		Platform: "CPU",
	}
	devices = append(devices, cpuDevice)

	return devices
}

// Helper function to estimate memory based on device name
func estimateDeviceMemory(deviceName string) uint64 {
	// Conservative estimates based on known Apple Silicon configurations
	switch {
	case contains(deviceName, "M3 Max"):
		return 36 * 1024 * 1024 * 1024 // 36GB
	case contains(deviceName, "M3 Pro"):
		return 18 * 1024 * 1024 * 1024 // 18GB
	case contains(deviceName, "M3"):
		return 16 * 1024 * 1024 * 1024 // 16GB
	case contains(deviceName, "M2 Max"):
		return 32 * 1024 * 1024 * 1024 // 32GB
	case contains(deviceName, "M2 Pro"):
		return 16 * 1024 * 1024 * 1024 // 16GB
	case contains(deviceName, "M2"):
		return 16 * 1024 * 1024 * 1024 // 16GB
	default:
		return 16 * 1024 * 1024 * 1024 // 16GB default
	}
}

// Simple string contains function
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

// GetBestDevice returns the highest performance available device
func GetBestDevice() Device {
	devices := DetectDevices()

	// Prefer Metal GPU over CPU
	for _, device := range devices {
		if device.Platform == "Metal" {
			return device
		}
	}

	// Fallback to CPU
	return devices[len(devices)-1]
}

// IsMetalAvailable checks if Metal is available on this system
func IsMetalAvailable() bool {
	devices := DetectDevices()
	for _, device := range devices {
		if device.Platform == "Metal" {
			return true
		}
	}
	return false
}
