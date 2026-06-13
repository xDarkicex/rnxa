//go:build darwin
// +build darwin

package mps

// mps_darwin.go — darwin implementation of the MPS backend using
// purego to load the pre-built libmps.dylib (see Makefile). No
// cgo: the .mm file is compiled out-of-band, the .dylib is
// loaded at runtime, and the Go side declares the function
// signatures natively.
//
// Why purego, not cgo: cgo's auto-naming of function return types
// in Objective-C++ files mishandles `void *` returns. The
// existing metal_darwin.go works around this by storing device/
// queue as `interface{}`; for the MPS backend the cleanest fix
// is to skip cgo entirely and use a pre-built library.

import (
	"errors"
	"fmt"
	"sync"

	"github.com/ebitengine/purego"
)

const libName = "libmps.dylib"

var (
	libOnce sync.Once
	lib     uintptr
	libErr  error
)

func loadLib() (uintptr, error) {
	libOnce.Do(func() {
		handle, err := purego.Dlopen(libName, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil || handle == 0 {
			libErr = fmt.Errorf("mps: failed to load %s: %v (build via `make` in this directory)", libName, err)
			return
		}
		lib = handle
	})
	return lib, libErr
}

// C function signatures exposed by libmps.dylib. Handles are raw
// intptr_t values (pointer values cast through intptr_t at the
// C ABI).
type (
	mpsDeviceNewFn     func() int64
	mpsDeviceReleaseFn func(int64)
	mpsQueueNewFn      func(int64) int64
	mpsQueueReleaseFn  func(int64)
	mpsMatmulFn        func(int64, int64, *float32, *float32, *float32, int64, int64, int64) int32
	mpsReLUFn          func(int64, int64, *float32, *float32, int64) int32
	mpsSigmoidFn       func(int64, int64, *float32, *float32, int64) int32
	mpsTanhFn          func(int64, int64, *float32, *float32, int64) int32
	mpsSoftmaxFn       func(int64, int64, *float32, *float32, int64) int32
)

var (
	fnMpsDeviceNew     mpsDeviceNewFn
	fnMpsDeviceRelease mpsDeviceReleaseFn
	fnMpsQueueNew      mpsQueueNewFn
	fnMpsQueueRelease  mpsQueueReleaseFn
	fnMpsMatmul        mpsMatmulFn
	fnMpsReLU          mpsReLUFn
	fnMpsSigmoid       mpsSigmoidFn
	fnMpsTanh          mpsTanhFn
	fnMpsSoftmax       mpsSoftmaxFn

	bindOnce sync.Once
	bindErr  error
)

func bindFns() error {
	bindOnce.Do(func() {
		handle, err := loadLib()
		if err != nil {
			bindErr = err
			return
		}
		bind := func(slot any, sym string) {
			cfn, e := purego.Dlsym(handle, sym)
			if e != nil || cfn == 0 {
				bindErr = fmt.Errorf("mps: symbol %q not found", sym)
				return
			}
			purego.RegisterFunc(slot, cfn)
		}
		bind(&fnMpsDeviceNew, "mps_device_new")
		bind(&fnMpsDeviceRelease, "mps_device_release")
		bind(&fnMpsQueueNew, "mps_queue_new")
		bind(&fnMpsQueueRelease, "mps_queue_release")
		bind(&fnMpsMatmul, "mps_matmul")
		bind(&fnMpsReLU, "mps_relu")
		bind(&fnMpsSigmoid, "mps_sigmoid")
		bind(&fnMpsTanh, "mps_tanh")
		bind(&fnMpsSoftmax, "mps_softmax")
	})
	return bindErr
}

func newMPSBackend() (*Backend, error) {
	if err := bindFns(); err != nil {
		return nil, err
	}
	dev := uintptr(fnMpsDeviceNew())
	if dev == 0 {
		return nil, errors.New("mps: no Metal device")
	}
	queue := uintptr(fnMpsQueueNew(int64(dev)))
	if queue == 0 {
		fnMpsDeviceRelease(int64(dev))
		return nil, errors.New("mps: failed to create command queue")
	}
	return &Backend{device: dev, queue: queue}, nil
}

func closeMPSBackend(b *Backend) error {
	if b == nil {
		return nil
	}
	if b.queue != 0 {
		fnMpsQueueRelease(int64(b.queue))
		b.queue = 0
	}
	if b.device != 0 {
		fnMpsDeviceRelease(int64(b.device))
		b.device = 0
	}
	return nil
}

func mpsMatmul(b *Backend, A, B []float32, M, N, K int) ([]float32, error) {
	if b == nil || b.device == 0 {
		return nil, errors.New("mps: backend not initialised")
	}
	if len(A) != M*K {
		return nil, fmt.Errorf("mps matmul: A length %d, expected %d", len(A), M*K)
	}
	if len(B) != K*N {
		return nil, fmt.Errorf("mps matmul: B length %d, expected %d", len(B), K*N)
	}
	aCopy := make([]float32, len(A))
	copy(aCopy, A)
	bCopy := make([]float32, len(B))
	copy(bCopy, B)
	c := make([]float32, M*N)
	rc := fnMpsMatmul(
		int64(b.device), int64(b.queue),
		&aCopy[0], &bCopy[0], &c[0],
		int64(M), int64(N), int64(K),
	)
	if rc != 0 {
		return nil, fmt.Errorf("mps matmul: kernel returned %d", rc)
	}
	return c, nil
}

func mpsReLU(b *Backend, X []float32) ([]float32, error) {
	return runUnary(b, "relu", X, fnMpsReLU)
}

func mpsSigmoid(b *Backend, X []float32) ([]float32, error) {
	return runUnary(b, "sigmoid", X, fnMpsSigmoid)
}

func mpsTanh(b *Backend, X []float32) ([]float32, error) {
	return runUnary(b, "tanh", X, fnMpsTanh)
}

func mpsSoftmax(b *Backend, X []float32) ([]float32, error) {
	if b == nil || b.device == 0 {
		return nil, errors.New("mps: backend not initialised")
	}
	xCopy := make([]float32, len(X))
	copy(xCopy, X)
	y := make([]float32, len(X))
	rc := fnMpsSoftmax(
		int64(b.device), int64(b.queue),
		&xCopy[0], &y[0],
		int64(len(X)),
	)
	if rc != 0 {
		return nil, fmt.Errorf("mps softmax: kernel returned %d", rc)
	}
	return y, nil
}

func runUnary(b *Backend, name string, X []float32,
	cFn func(int64, int64, *float32, *float32, int64) int32) ([]float32, error) {
	if b == nil || b.device == 0 {
		return nil, errors.New("mps: backend not initialised")
	}
	xCopy := make([]float32, len(X))
	copy(xCopy, X)
	y := make([]float32, len(X))
	rc := cFn(
		int64(b.device), int64(b.queue),
		&xCopy[0], &y[0],
		int64(len(X)),
	)
	if rc != 0 {
		return nil, fmt.Errorf("mps %s: kernel returned %d", name, rc)
	}
	return y, nil
}
