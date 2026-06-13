package rnxa

import (
	"math"
	"sync"
	"testing"
)

func TestReduceSum_FullReduction(t *testing.T) {
	x := NewTensor([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	got, err := reduceSum(x, AxisAll)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Data()) != 1 || math.Abs(got.Data()[0]-21) > 1e-9 {
		t.Errorf("full sum = %v, want [21]", got.Data())
	}
}

func TestReduceSum_AxisZero(t *testing.T) {
	// 2x3 tensor, axis=0 collapses rows -> [3] result of column sums.
	x := NewTensor([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	got, err := reduceSum(x, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{5, 7, 9}
	if !equalFloat64(got.Data(), want) {
		t.Errorf("axis=0 sum = %v, want %v", got.Data(), want)
	}
}

func TestReduceSum_AxisOne(t *testing.T) {
	// 2x3 tensor, axis=1 collapses cols -> [2] result of row sums.
	x := NewTensor([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	got, err := reduceSum(x, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{6, 15}
	if !equalFloat64(got.Data(), want) {
		t.Errorf("axis=1 sum = %v, want %v", got.Data(), want)
	}
}

func TestReduceSum_OutOfRange(t *testing.T) {
	x := NewTensor([]float64{1, 2, 3, 4}, 2, 2)
	if _, err := reduceSum(x, 2); err == nil {
		t.Error("expected error for axis=2 on 2D tensor, got nil")
	}
}

func TestReduceMean_AxisZero(t *testing.T) {
	x := NewTensor([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	got, err := reduceMean(x, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{2.5, 3.5, 4.5}
	if !equalFloat64(got.Data(), want) {
		t.Errorf("axis=0 mean = %v, want %v", got.Data(), want)
	}
}

func TestReduceMean_FullReduction(t *testing.T) {
	x := NewTensor([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	got, err := reduceMean(x, AxisAll)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got.Data()[0]-3.5) > 1e-9 {
		t.Errorf("full mean = %v, want 3.5", got.Data()[0])
	}
}

func TestReduceSum_Float32Path(t *testing.T) {
	x := NewTensorFromFloat32([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	got, err := reduceSum(x, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got.DType() != Float32 {
		t.Errorf("dtype = %v, want Float32", got.DType())
	}
	want := []float32{5, 7, 9}
	if !equalFloat32(got.Data32(), want) {
		t.Errorf("axis=0 f32 sum = %v, want %v", got.Data32(), want)
	}
}

// TestReduce_NoRace exercises the new non-mutating tensor data accessors
// concurrently. With the pre-fix lazy conversion, the first goroutine to
// touch a fresh float32 tensor would write t.data; the second would race.
func TestReduce_NoRace(t *testing.T) {
	x := NewTensorFromFloat32([]float32{1, 2, 3, 4}, 2, 2)

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
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

func TestCpuEngine_SumRespectsAxis(t *testing.T) {
	e := newCPUEngine()
	x := NewTensor([]float64{1, 2, 3, 4, 5, 6}, 2, 3)

	got, err := e.Sum(nil, x, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !equalFloat64(got.Data(), []float64{5, 7, 9}) {
		t.Errorf("Sum(axis=0) = %v, want [5 7 9]", got.Data())
	}

	got, err = e.Sum(nil, x, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !equalFloat64(got.Data(), []float64{6, 15}) {
		t.Errorf("Sum(axis=1) = %v, want [6 15]", got.Data())
	}
}

func TestCpuEngine_MeanRespectsAxis(t *testing.T) {
	e := newCPUEngine()
	x := NewTensor([]float64{2, 4, 6, 8}, 2, 2)

	got, err := e.Mean(nil, x, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !equalFloat64(got.Data(), []float64{4, 6}) {
		t.Errorf("Mean(axis=0) = %v, want [4 6]", got.Data())
	}
}

func equalFloat64(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-6 {
			return false
		}
	}
	return true
}

func equalFloat32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > 1e-5 {
			return false
		}
	}
	return true
}
