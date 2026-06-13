//go:build linux

package cuda

import (
	"math"
	"testing"
)

func TestCUDA_NewSucceeds(t *testing.T) {
	b, err := New()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	defer b.Close()
	if b == nil {
		t.Fatal("New returned nil backend without error")
	}
	if !b.HasHandles() {
		t.Fatal("HasHandles() false after successful New()")
	}
}

func TestCUDA_MatMulMatches(t *testing.T) {
	b, err := New()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	defer b.Close()

	// 2x3 * 3x2 = 2x2. A row-major, B row-major.
	A := []float32{1, 2, 3, 4, 5, 6}
	B := []float32{7, 8, 9, 10, 11, 12}
	expected := []float32{58, 64, 139, 154}

	c, err := b.MatMul(A, B, 2, 2, 3)
	if err != nil {
		t.Fatalf("MatMul: %v", err)
	}
	if len(c) != len(expected) {
		t.Fatalf("len = %d, want %d", len(c), len(expected))
	}
	for i, v := range c {
		if math.Abs(float64(v-expected[i])) > 1e-2 {
			t.Errorf("c[%d] = %v, want %v (FP32 tolerance)", i, v, expected[i])
		}
	}
}

func TestCUDA_VectorAdd(t *testing.T) {
	b, err := New()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	defer b.Close()

	a := []float32{1, 2, 3, 4, 5, 6}
	bb := []float32{10, 20, 30, 40, 50, 60}
	want := []float32{11, 22, 33, 44, 55, 66}
	got, err := b.Eltwise(a, bb, 6, 0) // op 0 = add
	if err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("eltwise add[%d] = %v, want %v", i, got[i], want[i])
		}
	}

	// sub: bb - a
	got, err = b.Eltwise(bb, a, 6, 1) // op 1 = sub
	if err != nil {
		t.Fatal(err)
	}
	wantSub := []float32{9, 18, 27, 36, 45, 54}
	for i := range got {
		if got[i] != wantSub[i] {
			t.Errorf("eltwise sub[%d] = %v, want %v", i, got[i], wantSub[i])
		}
	}

	// mul: a * bb
	got, err = b.Eltwise(a, bb, 6, 2) // op 2 = mul
	if err != nil {
		t.Fatal(err)
	}
	wantMul := []float32{10, 40, 90, 160, 250, 360}
	for i := range got {
		if got[i] != wantMul[i] {
			t.Errorf("eltwise mul[%d] = %v, want %v", i, got[i], wantMul[i])
		}
	}
}

func TestCUDA_ReLU(t *testing.T) {
	b, err := New()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	defer b.Close()

	in := []float32{-2, -1, 0, 1, 2}
	want := []float32{0, 0, 0, 1, 2}
	got, err := b.ReLU(in)
	if err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("ReLU[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestCUDA_SoftmaxSumsToOne(t *testing.T) {
	b, err := New()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	defer b.Close()

	in := []float32{1.0, 2.0, 3.0, 4.0}
	got, err := b.Softmax(in)
	if err != nil {
		t.Fatal(err)
	}
	var sum float32
	for _, v := range got {
		sum += v
	}
	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("softmax sum = %v, want ~1.0", sum)
	}
	// Softmax is monotonically increasing on the input.
	for i := 1; i < len(got); i++ {
		if got[i] <= got[i-1] {
			t.Errorf("softmax not monotonic: got[%d]=%v <= got[%d]=%v", i, got[i], i-1, got[i-1])
		}
	}
}
