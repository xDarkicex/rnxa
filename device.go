package rnxa

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework Foundation
#include <Metal/Metal.h>
#include <stdlib.h>

const char* get_device_name(void* device) {
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    return [mtlDevice.name UTF8String];
}

int get_device_cores(void* device) {
    // Approximate GPU cores based on device name
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    NSString* name = mtlDevice.name;
    if ([name containsString:@"M3 Max"]) return 40;
    if ([name containsString:@"M3 Pro"]) return 18;
    if ([name containsString:@"M3"]) return 10;
    if ([name containsString:@"M2 Max"]) return 38;
    if ([name containsString:@"M2 Pro"]) return 19;
    if ([name containsString:@"M2"]) return 10;
    return 8; // Default estimate
}
*/
import "C"
import (
	"unsafe"
)

// DetectDevices discovers available Metal devices
func DetectDevices() []Device {
	var devices []Device

	// Try to create Metal device
	metalDevice := C.MTLCreateSystemDefaultDevice()
	if metalDevice != nil {
		defer C.CFRelease(metalDevice)

		name := C.GoString(C.get_device_name(unsafe.Pointer(metalDevice)))
		cores := int(C.get_device_cores(unsafe.Pointer(metalDevice)))
		memory := uint64(16) * 1024 * 1024 * 1024 // Default 16GB unified memory

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
