package alloc_test

import (
	"runtime"
	"sync"
	"testing"

	"github.com/xDarkicex/rnxa/alloc"
)

func TestFloat64_Valid(t *testing.T) {
	s := alloc.Float64(1024)
	if len(s) != 1024 {
		t.Fatalf("len = %d, want 1024", len(s))
	}
	for i, v := range s {
		if v != 0 {
			t.Fatalf("s[%d] = %v, want 0", i, v)
		}
	}
	for i := range s {
		s[i] = float64(i)
	}
	for i, v := range s {
		if v != float64(i) {
			t.Fatalf("s[%d] = %v, want %v", i, v, i)
		}
	}
}

func TestFloat32_Valid(t *testing.T) {
	s := alloc.Float32(512)
	if len(s) != 512 {
		t.Fatalf("len = %d, want 512", len(s))
	}
	for i, v := range s {
		if v != 0 {
			t.Fatalf("s[%d] = %v, want 0", i, v)
		}
	}
}

func TestNonPositive(t *testing.T) {
	if s := alloc.Float64(0); s != nil {
		t.Errorf("Float64(0) = %v, want nil", s)
	}
	if s := alloc.Float32(-1); s != nil {
		t.Errorf("Float32(-1) = %v, want nil", s)
	}
}

func TestStress_Concurrent(t *testing.T) {
	const goroutines = 16
	const perGoroutine = 1000

	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < perGoroutine; i++ {
				s := alloc.Float64(128)
				if len(s) != 128 {
					t.Errorf("len = %d, want 128", len(s))
					return
				}
				s[0] = 1.0
				s[127] = -1.0
				runtime.KeepAlive(s)
			}
		}()
	}
	wg.Wait()
}

func TestFinalize_Idempotent(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Finalize panicked: %v", r)
		}
	}()
	alloc.Finalize()
	alloc.Finalize()
}

func TestFree_RoundTrip(t *testing.T) {
	s := alloc.Float64(64)
	if len(s) != 64 {
		t.Fatalf("len = %d, want 64", len(s))
	}
	for i := range s {
		s[i] = float64(i + 1)
	}
	alloc.Free(s)

	s2 := alloc.Float64(64)
	for i, v := range s2 {
		if v != 0 {
			t.Errorf("realloc s2[%d] = %v, want 0 (re-zeroed)", i, v)
		}
	}
	alloc.Free(s2)
}

func TestFree_NonAllocated(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Free on non-allocated slice panicked: %v", r)
		}
	}()
	s := make([]float64, 16)
	alloc.Free(s) // no-op
}
