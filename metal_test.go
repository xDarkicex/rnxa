package rnxa

import (
	"context"
	"math"
	"testing"
)

func TestDeviceDetection(t *testing.T) {
	devices := DetectDevices()

	if len(devices) == 0 {
		t.Fatal("No devices detected")
	}

	// Should always have CPU fallback
	foundCPU := false
	for _, device := range devices {
		if device.Platform == "CPU" {
			foundCPU = true
			break
		}
	}

	if !foundCPU {
		t.Error("CPU fallback device not found")
	}

	t.Logf("Found %d devices:", len(devices))
	for i, device := range devices {
		t.Logf("  Device %d: %s (%s) - %d cores, %.1fGB memory",
			i, device.Name, device.Platform, device.Cores, float64(device.Memory)/1e9)
	}
}

func TestEngineCreation(t *testing.T) {
	engine, err := NewEngine()
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Close()

	if !engine.Available() {
		t.Error("Engine reports as not available")
	}

	device := engine.Device()
	t.Logf("Created engine for device: %s (%s)", device.Name, device.Platform)
}

func TestMatrixMultiplication(t *testing.T) {
	engine, err := NewEngine()
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Close()

	// Test small matrix multiplication: [2x3] × [3x2] = [2x2]
	A := NewTensor([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	B := NewTensor([]float64{7, 8, 9, 10, 11, 12}, 3, 2)

	ctx := context.Background()
	C, err := engine.MatMul(ctx, A, B)
	if err != nil {
		t.Fatalf("MatMul failed: %v", err)
	}

	expected := []float64{58, 64, 139, 154} // Expected result

	if len(C.Data()) != len(expected) {
		t.Fatalf("Result size mismatch: got %d, expected %d", len(C.Data()), len(expected))
	}

	for i, got := range C.Data() {
		if math.Abs(got-expected[i]) > 1e-5 {
			t.Errorf("Result[%d] = %f, expected %f", i, got, expected[i])
		}
	}

	t.Logf("MatMul test passed on %s", engine.Device().Platform)
}

func TestActivationFunctions(t *testing.T) {
	engine, err := NewEngine()
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Close()

	ctx := context.Background()

	// Test data: [-2, -1, 0, 1, 2]
	input := NewTensor([]float64{-2, -1, 0, 1, 2})

	// Test ReLU
	relu_result, err := engine.ReLU(ctx, input)
	if err != nil {
		t.Errorf("ReLU failed: %v", err)
	} else {
		expected_relu := []float64{0, 0, 0, 1, 2}
		for i, got := range relu_result.Data() {
			if math.Abs(got-expected_relu[i]) > 1e-5 {
				t.Errorf("ReLU[%d] = %f, expected %f", i, got, expected_relu[i])
			}
		}
		t.Logf("ReLU test passed on %s", engine.Device().Platform)
	}

	// Test Sigmoid
	sigmoid_result, err := engine.Sigmoid(ctx, input)
	if err != nil {
		t.Errorf("Sigmoid failed: %v", err)
	} else {
		// Sigmoid should produce values in (0,1)
		for i, got := range sigmoid_result.Data() {
			if got <= 0 || got >= 1 {
				t.Errorf("Sigmoid[%d] = %f, should be in (0,1)", i, got)
			}
		}
		t.Logf("Sigmoid test passed on %s", engine.Device().Platform)
	}

	// Test Tanh
	tanh_result, err := engine.Tanh(ctx, input)
	if err != nil {
		t.Errorf("Tanh failed: %v", err)
	} else {
		// Tanh should produce values in (-1,1)
		for i, got := range tanh_result.Data() {
			if got <= -1 || got >= 1 {
				t.Errorf("Tanh[%d] = %f, should be in (-1,1)", i, got)
			}
		}
		t.Logf("Tanh test passed on %s", engine.Device().Platform)
	}
}

func TestVectorOperations(t *testing.T) {
	engine, err := NewEngine()
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Close()

	ctx := context.Background()

	A := NewTensor([]float64{1, 2, 3, 4})
	B := NewTensor([]float64{5, 6, 7, 8})

	// Test VectorAdd
	add_result, err := engine.VectorAdd(ctx, A, B)
	if err != nil {
		t.Errorf("VectorAdd failed: %v", err)
	} else {
		expected := []float64{6, 8, 10, 12}
		for i, got := range add_result.Data() {
			if math.Abs(got-expected[i]) > 1e-5 {
				t.Errorf("VectorAdd[%d] = %f, expected %f", i, got, expected[i])
			}
		}
		t.Logf("VectorAdd test passed on %s", engine.Device().Platform)
	}

	// Test VectorSub
	sub_result, err := engine.VectorSub(ctx, A, B)
	if err != nil {
		t.Errorf("VectorSub failed: %v", err)
	} else {
		expected := []float64{-4, -4, -4, -4}
		for i, got := range sub_result.Data() {
			if math.Abs(got-expected[i]) > 1e-5 {
				t.Errorf("VectorSub[%d] = %f, expected %f", i, got, expected[i])
			}
		}
		t.Logf("VectorSub test passed on %s", engine.Device().Platform)
	}
}

// Benchmark tests for performance validation
func BenchmarkMatMulSmall(b *testing.B) {
	engine, _ := NewEngine()
	defer engine.Close()

	A := NewTensor(make([]float64, 32*32), 32, 32)
	B := NewTensor(make([]float64, 32*32), 32, 32)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.MatMul(ctx, A, B)
	}
}

func BenchmarkMatMulMedium(b *testing.B) {
	engine, _ := NewEngine()
	defer engine.Close()

	A := NewTensor(make([]float64, 128*128), 128, 128)
	B := NewTensor(make([]float64, 128*128), 128, 128)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.MatMul(ctx, A, B)
	}
}
