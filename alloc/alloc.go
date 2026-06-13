// Package alloc provides off-heap tensor-data allocation backed by the
// xDarkicex/memory package. The intent is to keep the bulk of every
// tensor's data (activations, weights, intermediate results) out of
// the Go GC's reach so that batched inference and training don't pay
// stop-the-world pauses proportional to working-set size.
//
// The backing allocator is a memory.ShardedFreeList — lock-free
// Treiber stacks sharded by goroutine, with per-object Deallocate so
// callers may return slots to the freelist for reuse. Slices are
// bucketed by size (rounded up to the next power of two) so a small
// handful of freelists serves the full range of allocation sizes.
//
// The slice header (the {data, len, cap} struct) still lives on the Go
// heap, so the GC can trace the slice itself; only the underlying byte
// buffer is mmap'd. This means tensors can be freely passed to cgo
// kernels (Metal, CUDA) without violating the cgo pointer rules.
//
// On any allocation failure the helpers fall back to the Go heap.
package alloc

import (
	"sync"
	"unsafe"

	"github.com/xDarkicex/memory"
)

const (
	minSlotSize     = 48 // ShardedFreeList minimum (Hyaline padded to 48)
	defaultPoolSize = 16 * 1024 * 1024
)

var (
	freelists sync.Map // map[uint64]*memory.ShardedFreeList, keyed by slot size
	slotOwner sync.Map // map[uintptr]slotInfo, used by Free
)

type slotInfo struct {
	fl       *memory.ShardedFreeList
	slotSize uint64
}

func bucketFor(size uint64) uint64 {
	if size <= minSlotSize {
		return minSlotSize
	}
	p := uint64(minSlotSize)
	for p < size {
		p <<= 1
	}
	return p
}

func getFreelist(slotSize uint64) *memory.ShardedFreeList {
	if v, ok := freelists.Load(slotSize); ok {
		return v.(*memory.ShardedFreeList)
	}
	cfg := memory.DefaultFreeListConfig()
	cfg.SlotSize = slotSize
	cfg.PoolSize = defaultPoolSize
	fl, err := memory.NewShardedFreeList(cfg, 0)
	if err != nil {
		return nil
	}
	actual, _ := freelists.LoadOrStore(slotSize, fl)
	return actual.(*memory.ShardedFreeList)
}

// Float64 returns a float64 slice of length n whose backing storage
// lives in an off-heap freelist. The slice is zero-initialized
// (ShardedFreeList reuses slots, so leftover data is wiped on hand-out).
// Use [Free] to return the slice to the freelist for reuse.
func Float64(n int) []float64 {
	if n <= 0 {
		return nil
	}
	slotSize := bucketFor(uint64(n) * 8)
	fl := getFreelist(slotSize)
	if fl != nil {
		if buf, err := fl.Allocate(); err == nil {
			addr := uintptr(unsafe.Pointer(&buf[0]))
			slotOwner.Store(addr, slotInfo{fl: fl, slotSize: slotSize})
			s := unsafe.Slice((*float64)(unsafe.Pointer(&buf[0])), n)
			clear(s)
			return s
		}
	}
	return make([]float64, n)
}

// Float32 returns a float32 slice of length n, zero-initialized.
// See [Float64] for semantics.
func Float32(n int) []float32 {
	if n <= 0 {
		return nil
	}
	slotSize := bucketFor(uint64(n) * 4)
	fl := getFreelist(slotSize)
	if fl != nil {
		if buf, err := fl.Allocate(); err == nil {
			addr := uintptr(unsafe.Pointer(&buf[0]))
			slotOwner.Store(addr, slotInfo{fl: fl, slotSize: slotSize})
			s := unsafe.Slice((*float32)(unsafe.Pointer(&buf[0])), n)
			clear(s)
			return s
		}
	}
	return make([]float32, n)
}

// Int returns an int slice of length n. Used for shape/stride arrays.
// Stays on the Go heap (size class is tiny, GC overhead is negligible).
func Int(n int) []int {
	if n <= 0 {
		return nil
	}
	return make([]int, n)
}

// Free returns a slice previously obtained from [Float64] or [Float32]
// to its freelist for reuse. Calling Free on a slice that was not
// produced by this package, or that has already been freed, is a
// no-op. After Free, the slice must not be used; the underlying
// memory may be handed out to a subsequent allocation.
func Free[T any](s []T) {
	if len(s) == 0 {
		return
	}
	addr := uintptr(unsafe.Pointer(&s[0]))
	v, ok := slotOwner.LoadAndDelete(addr)
	if !ok {
		return
	}
	info := v.(slotInfo)
	slot := unsafe.Slice((*byte)(unsafe.Pointer(addr)), info.slotSize)
	_ = info.fl.Deallocate(slot)
}

// Finalize releases every off-heap freelist this package has created.
// Idempotent. The OS reclaims mmap'd pages on process exit even
// without an explicit Finalize.
func Finalize() {
	freelists.Range(func(k, v any) bool {
		_ = v.(*memory.ShardedFreeList).Free()
		freelists.Delete(k)
		return true
	})
	slotOwner.Clear()
}
