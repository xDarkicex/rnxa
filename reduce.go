package rnxa

import (
	"fmt"
	"math"
)

// AxisAll is the sentinel value for "reduce every axis into a single scalar".
// Any negative axis value has the same meaning for forward compatibility.
const AxisAll = -1

// reduceSum returns the sum of x along the given axis.
//
//   - axis == AxisAll (or any negative value): reduce every axis; result is a
//     scalar wrapped in a 1-element 1D tensor (preserves the pre-fix behavior
//     of Sum/Mean so callers that never cared about axis still get a scalar).
//   - 0 <= axis < ndim: reduce that single axis; the result has the same
//     number of dimensions as x with axis removed (PyTorch keepDim=false).
//   - otherwise: an error is returned.
func reduceSum(x *Tensor, axis int) (*Tensor, error) {
	shape := x.Shape()
	ndim := len(shape)
	if axis < 0 {
		return fullSum(x)
	}
	if axis >= ndim {
		return nil, fmt.Errorf("sum: axis %d out of range for %dD tensor", axis, ndim)
	}

	// Build output shape by removing the reduced axis.
	outShape := make([]int, 0, ndim-1)
	for i, d := range shape {
		if i != axis {
			outShape = append(outShape, d)
		}
	}
	if len(outShape) == 0 {
		outShape = []int{1}
	}

	reducedSize := shape[axis]

	switch x.DType() {
	case Float32:
		result := ZerosFloat32(outShape...)
		src := x.float32Data()
		dst := result.float32Data()
		accumulateAxis(src, dst, shape, outShape, axis, reducedSize)
		return result, nil
	default:
		result := Zeros(outShape...)
		src := x.float64Data()
		dst := result.float64Data()
		accumulateAxis(src, dst, shape, outShape, axis, reducedSize)
		return result, nil
	}
}

// reduceMean returns the mean of x along the given axis. It calls reduceSum
// and divides by the size of the reduced dimension.
func reduceMean(x *Tensor, axis int) (*Tensor, error) {
	sum, err := reduceSum(x, axis)
	if err != nil {
		return nil, err
	}
	var n int
	if axis == AxisAll || axis < 0 {
		n = x.Size()
	} else {		shape := x.Shape()
		if axis < 0 || axis >= len(shape) {
			return nil, fmt.Errorf("mean: axis %d out of range for %dD tensor", axis, len(shape))
		}
		n = shape[axis]
	}
	if n == 0 {
		return nil, fmt.Errorf("mean: reduced axis has size 0")
	}
	switch sum.DType() {
	case Float32:
		dst := sum.float32Data()
		scale := float32(1.0 / float64(n))
		for i := range dst {
			dst[i] *= scale
		}
	default:
		dst := sum.float64Data()
		scale := 1.0 / float64(n)
		for i := range dst {
			dst[i] *= scale
		}
	}
	return sum, nil
}

// Sum reduces x along the given axis. Public wrapper around the
// internal reduceSum used by engine implementations that don't have
// their own native reduction kernel.
func Sum(x *Tensor, axis int) (*Tensor, error) { return reduceSum(x, axis) }

// Mean reduces x along the given axis and divides by the reduced size.
func Mean(x *Tensor, axis int) (*Tensor, error) { return reduceMean(x, axis) }

// fullSum reduces every axis into a scalar. Preserves the original Sum/Mean
// behavior for callers that pass axis=AxisAll.
func fullSum(x *Tensor) (*Tensor, error) {
	switch x.DType() {
	case Float32:
		src := x.float32Data()
		var s float32
		for _, v := range src {
			s += v
		}
		return NewTensorFromFloat32([]float32{s}), nil
	default:
		src := x.float64Data()
		s := 0.0
		for _, v := range src {
			s += v
		}
		return NewTensor([]float64{s}), nil
	}
}

// accumulateAxis walks every flat index in src and adds its value into the
// matching output cell, dropping the contribution of `axis` from the output
// index. src and dst are flat row-major slices. Shape/dstShape describe the
// corresponding layouts.
func accumulateAxis[T float32 | float64](src, dst []T, shape, dstShape []int, axis, reducedSize int) {
	ndim := len(shape)
	// Decompose the destination flat index back into a multi-index by
	// multiplying by dstShape strides; for the reduced axis, the inner loop
	// sweeps every value.
	dstStrides := make([]int, len(dstShape))
	if len(dstShape) > 0 {
		dstStrides[len(dstStrides)-1] = 1
		for i := len(dstStrides) - 2; i >= 0; i-- {
			dstStrides[i] = dstStrides[i+1] * dstShape[i+1]
		}
	}

	// Iterate output cells; for each, sum over the reduced axis.
	total := 1
	for _, d := range shape {
		total *= d
	}
	// Position in src as we walk. We rebuild the multi-index for every src
	// element rather than incrementing a single counter, because the
	// reduced-axis stride isn't 1 in general.
	var idxBuf [8]int
	if ndim > len(idxBuf) {
		// Fall back to heap for higher-rank tensors; rare.
		idxBuf = [8]int{}
		idx := make([]int, ndim)
		_ = idx
	}
	idx := idxBuf[:ndim]

	for i := 0; i < total; i++ {
		// Convert flat i to multi-index using row-major strides.
		rem := i
		for d := ndim - 1; d >= 0; d-- {
			idx[d] = rem % shape[d]
			rem /= shape[d]
		}
		// Compute destination flat index (skip axis).
		dstIdx := 0
		for d, oi := 0, 0; d < ndim; d++ {
			if d == axis {
				continue
			}
			dstIdx += idx[d] * dstStrides[oi]
			oi++
		}
		dst[dstIdx] += src[i]
	}
}

// Suppress unused-import linter on math; kept for future reductions.
var _ = math.Inf
