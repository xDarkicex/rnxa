package rnxa

import (
	"context"
	"fmt"
)

func matMulFloat32CPU(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	_ = ctx

	if len(A.Shape()) != 2 || len(B.Shape()) != 2 {
		return nil, fmt.Errorf("MatMul requires 2D tensors")
	}

	M, K1 := A.Shape()[0], A.Shape()[1]
	K2, N := B.Shape()[0], B.Shape()[1]
	if K1 != K2 {
		return nil, fmt.Errorf("incompatible matrix dimensions: (%d,%d) × (%d,%d)", M, K1, K2, N)
	}

	AData := A.float32Data()
	BData := B.float32Data()
	result := ZerosFloat32(M, N)
	resultData := result.float32Data()
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float32
			for k := 0; k < K1; k++ {
				sum += AData[i*K1+k] * BData[k*N+j]
			}
			resultData[i*N+j] = sum
		}
	}

	return result, nil
}
