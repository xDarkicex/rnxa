package rnxa

import (
	"math"
	"sync"
	"testing"
)

// TestTensor_NoLazyMutation verifies that float64Data() on a float32 tensor
// does not silently cache a converted copy. The pre-fix behavior would
// return a snapshot that becomes stale if data32 is later modified.
func TestTensor_NoLazyMutation(t *testing.T) {
	x := NewTensorFromFloat32([]float32{1, 2, 3, 4}, 2, 2)

	// Pre-fix: first call to float64Data() would allocate and cache t.data.
	// Post-fix: it returns nil, signalling the dtype mismatch.
	if got := x.float64Data(); got != nil {
		t.Errorf("float64Data() on a Float32 tensor should be nil, got %v", got)
	}

	// Mutate the underlying float32 buffer.
	x.data32[0] = 99

	// Pre-fix would have returned a stale [1 2 3 4]; post-fix still nil.
	if got := x.float64Data(); got != nil {
		t.Errorf("float64Data() should remain nil after data32 mutation, got %v", got)
	}
}

// TestTensor_ConvertAllocatesAndIsFresh verifies the public conversion helper
// returns a freshly-allocated slice (so callers may safely mutate it without
// affecting the source tensor).
func TestTensor_ConvertAllocatesAndIsFresh(t *testing.T) {
	x := NewTensorFromFloat32([]float32{1, 2, 3, 4}, 2, 2)
	converted := toFloat64(x)
	if len(converted) != 4 {
		t.Fatalf("converted length = %d, want 4", len(converted))
	}
	converted[0] = 999
	if x.data32[0] != 1 {
		t.Errorf("mutating the converted slice should not affect the source tensor; data32[0] = %v, want 1", x.data32[0])
	}
}

// TestTensor_ConcurrentDataAccess is the regression test for the data race
// in the old lazy conversion. Run with -race; pre-fix this would fail.
func TestTensor_ConcurrentDataAccess(t *testing.T) {
	x := NewTensorFromFloat32(make([]float32, 100), 10, 10)

	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			_ = toFloat64(x)
		}()
		go func() {
			defer wg.Done()
			_ = toFloat32(x)
		}()
	}
	wg.Wait()
}

func TestTensor_ReshapePreservesData(t *testing.T) {
	x := NewTensor([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	r := x.Reshape(3, 2)
	if r.Size() != 6 {
		t.Errorf("reshaped size = %d, want 6", r.Size())
	}
	for i, v := range r.Data() {
		if math.Abs(v-x.Data()[i]) > 1e-9 {
			t.Errorf("reshape mismatch at %d: %v vs %v", i, v, x.Data()[i])
		}
	}
}
