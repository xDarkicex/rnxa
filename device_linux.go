//go:build linux
// +build linux

package rnxa

import (
	"bytes"
	"os/exec"
	"strconv"
	"strings"
)

// DetectDevices discovers available compute devices on Linux. The
// primary GPU path is NVIDIA CUDA, probed by shelling out to
// `nvidia-smi`. A CPU fallback is always appended so the
// dispatcher never returns an empty list.
//
// Why nvidia-smi and not the CUDA driver API via purego: the
// "real" long-term answer is to dlopen libcuda.so.1 and call
// cuInit / cuDeviceGetCount. That's a follow-up — first cut
// uses nvidia-smi because it's already a runtime requirement
// for anyone who has an NVIDIA driver loaded, and the parse is
// trivial.
//
// When nvidia-smi is missing, fails, or reports zero devices,
// the CUDA device is silently omitted from the list. The CPU
// fallback still appears. The dispatcher in NewEngine will
// skip the CUDA case and fall through to CPU.
func DetectDevices() []Device {
	var devices []Device

	if gpus := probeNvidiaGPUs(); len(gpus) > 0 {
		devices = append(devices, gpus...)
	}

	// CPU fallback (always last).
	devices = append(devices, Device{
		ID:       len(devices),
		Name:     "CPU",
		Type:     CPU,
		Memory:   8 * 1024 * 1024 * 1024,
		Cores:    8,
		Platform: "CPU",
	})

	return devices
}

// probeNvidiaGPUs shells out to nvidia-smi and parses the
// output. Returns an empty slice (not an error) if nvidia-smi is
// missing, fails, or reports no GPUs. The ID column is mapped
// 1:1 to Device.ID so callers can later select by index via
// NewEngineWithDevice.
//
// `nvidia-smi --query-gpu=index,name,memory.total
//   --format=csv,noheader,nounits`
// emits one line per GPU like:
//   0, NVIDIA GeForce RTX 4090, 24564
func probeNvidiaGPUs() []Device {
	cmd := exec.Command("nvidia-smi",
		"--query-gpu=index,name,memory.total",
		"--format=csv,noheader,nounits")
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		// nvidia-smi missing, or no driver loaded, or the
		// user has no NVIDIA hardware. Any of those → no
		// CUDA device, fall through to CPU only.
		return nil
	}

	var gpus []Device
	for _, line := range strings.Split(out.String(), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// Split on the first two commas only; the name may
		// contain commas in pathological cases (e.g. "Foo,
		// Inc. RTX 4090"), though that's vanishingly rare.
		parts := strings.SplitN(line, ",", 3)
		if len(parts) < 3 {
			continue
		}
		idx, err := strconv.Atoi(strings.TrimSpace(parts[0]))
		if err != nil {
			continue
		}
		name := strings.TrimSpace(parts[1])
		memMiB, err := strconv.ParseInt(strings.TrimSpace(parts[2]), 10, 64)
		if err != nil {
			// memory.total is unparseable; still surface
			// the device, just with zero memory so the
			// dispatcher can pick it. The CUDA shim will
			// report its own memory at New() time.
			memMiB = 0
		}

		gpus = append(gpus, Device{
			ID:       idx,
			Name:     name,
			Type:     GPU,
			Memory:   uint64(memMiB) * 1024 * 1024,
			Cores:    0, // SM count isn't surfaced by nvidia-smi
			Platform: "CUDA",
		})
	}
	return gpus
}

func GetBestDevice() Device {
	// First CUDA device if present, else CPU.
	devices := DetectDevices()
	for _, d := range devices {
		if d.Platform == "CUDA" {
			return d
		}
	}
	return devices[len(devices)-1]
}

func IsMetalAvailable() bool { return false }
