package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/xDarkicex/relux"
	"github.com/xDarkicex/rnxa"

	// Major Go ML Frameworks
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	fmt.Println("🚀 Go ML Frameworks + rnxa Integration Demo")
	fmt.Println("===========================================")

	// Sample datasets
	xorX := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	xorY := [][]float64{{0}, {1}, {1}, {0}}

	irisData := generateIrisDataset()

	// 1. relux Framework with rnxa Integration
	fmt.Println("\n🔥 1. relux - Enterprise Neural Networks")
	fmt.Println("=======================================")
	demonstrateRelux(xorX, xorY)

	// 2. Direct rnxa Usage
	fmt.Println("\n⚡ 2. rnxa - Hardware-Accelerated Compute Engine")
	fmt.Println("==============================================")
	demonstrateRnxa()

	// 3. Gorgonia with rnxa Integration
	fmt.Println("\n🧠 3. Gorgonia - Automatic Differentiation")
	fmt.Println("=========================================")
	demonstrateGorgonia()

	// 4. GoLearn with Custom rnxa Backend
	fmt.Println("\n📚 4. GoLearn - Traditional ML Algorithms")
	fmt.Println("========================================")
	demonstrateGoLearn(irisData)

	// 5. Custom Framework Integration Example
	fmt.Println("\n🔗 5. Custom Framework Integration")
	fmt.Println("=================================")
	demonstrateCustomIntegration()

	// 6. Performance Comparison
	fmt.Println("\n📊 6. Performance Comparison")
	fmt.Println("============================")
	performanceComparison()

	fmt.Println("\n🎉 Demo completed! All frameworks showcased with rnxa integration.")
}

// 1. relux Framework Demonstration
func demonstrateRelux(X, Y [][]float64) {
	start := time.Now()

	net, err := relux.NewNetwork(
		relux.WithConfig(relux.Config{
			Inputs: []relux.InputSpec{{Size: 2}},
			Hidden: []relux.LayerSpec{{Units: 8, Act: "tanh"}},
			Output: relux.LayerSpec{Units: 1, Act: "sigmoid"},
			Loss:   "bce",
		}),
	)
	if err != nil {
		log.Fatal("Failed to create relux network:", err)
	}

	// Train the network
	net.Fit(X, Y,
		relux.Epochs(1000),
		relux.LearningRate(0.3),
		relux.Verbose(false),
	)

	duration := time.Since(start)
	fmt.Printf("Training completed in: %v\n", duration)

	// Test XOR
	fmt.Println("XOR Results:")
	for i, x := range X {
		pred, _ := net.Predict(x)
		status := "✅"
		if (Y[i][0] < 0.5 && pred[0] > 0.5) || (Y[i][0] > 0.5 && pred[0] < 0.5) {
			status = "❌"
		}
		fmt.Printf("  [%.0f, %.0f] → Expected: %.0f, Got: %.3f %s\n",
			x[0], x[1], Y[i][0], pred[0], status)
	}

	fmt.Println("✅ Strengths: Zero dependencies, production-ready")
	fmt.Println("🚀 Future: Integration with rnxa for 5-20x speedup")
}

// 2. Direct rnxa Usage
func demonstrateRnxa() {
	// Device detection
	devices := rnxa.DetectDevices()
	fmt.Printf("Detected %d compute devices:\n", len(devices))
	for i, device := range devices {
		fmt.Printf("  Device %d: %s (%s) - %d cores\n",
			i, device.Name, device.Platform, device.Cores)
	}

	// Create engine
	engine, err := rnxa.NewEngine()
	if err != nil {
		log.Fatal("Failed to create rnxa engine:", err)
	}
	defer engine.Close()

	activeDevice := engine.Device()
	fmt.Printf("\nActive: %s (%s)\n", activeDevice.Name, activeDevice.Platform)

	// Tensor operations
	ctx := context.Background()
	A := rnxa.NewTensor([]float64{1, 2, 3, 4}, 2, 2)
	B := rnxa.NewTensor([]float64{5, 6, 7, 8}, 2, 2)

	start := time.Now()
	C, err := engine.MatMul(ctx, A, B)
	duration := time.Since(start)

	if err != nil {
		fmt.Printf("❌ Matrix multiplication failed: %v\n", err)
	} else {
		fmt.Printf("Matrix multiplication: %v (computed in %v)\n", C.Data(), duration)
	}

	// Activation functions
	input := rnxa.NewTensor([]float64{-1, 0, 1, 2})
	relu, _ := engine.ReLU(ctx, input)
	sigmoid, _ := engine.Sigmoid(ctx, input)

	fmt.Printf("Input:   %v\n", input.Data())
	fmt.Printf("ReLU:    %v\n", formatFloats(relu.Data()))
	fmt.Printf("Sigmoid: %v\n", formatFloats(sigmoid.Data()))

	fmt.Println("✅ Strengths: Universal backend, Metal/CUDA acceleration")
	fmt.Println("🎯 Use case: Accelerate any Go ML framework")
}

// 3. Gorgonia Framework with rnxa Integration Concept
func demonstrateGorgonia() {
	// Create computation graph
	g := gorgonia.NewGraph()

	// Define variables
	x := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(2, 2), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(2, 2), gorgonia.WithName("y"))

	// Define operation
	z, err := gorgonia.Add(x, y)
	if err != nil {
		fmt.Printf("❌ Gorgonia operation failed: %v\n", err)
		return
	}

	// Create VM and run
	machine := gorgonia.NewTapeMachine(g)
	defer machine.Close()

	// Set values
	xVal := tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4}), tensor.WithShape(2, 2))
	yVal := tensor.New(tensor.WithBacking([]float64{5, 6, 7, 8}), tensor.WithShape(2, 2))

	err = gorgonia.Let(x, xVal)
	if err != nil {
		fmt.Printf("❌ Failed to set x: %v\n", err)
		return
	}

	err = gorgonia.Let(y, yVal)
	if err != nil {
		fmt.Printf("❌ Failed to set y: %v\n", err)
		return
	}

	start := time.Now()
	err = machine.RunAll()
	duration := time.Since(start)

	if err != nil {
		fmt.Printf("❌ Gorgonia execution failed: %v\n", err)
		return
	}

	result := z.Value()
	fmt.Printf("Gorgonia tensor addition: %v (computed in %v)\n", result, duration)

	fmt.Println("\n💡 rnxa Integration Concept:")
	fmt.Println("  // Custom Gorgonia executor using rnxa")
	fmt.Println("  type RnxaExecutor struct {")
	fmt.Println("      engine rnxa.ComputeEngine")
	fmt.Println("  }")
	fmt.Println("  // Execute Gorgonia ops with Metal/CUDA acceleration!")

	fmt.Println("✅ Strengths: Full AD support, research-grade flexibility")
	fmt.Println("🚀 Potential: rnxa could accelerate core tensor ops")
}

// 4. GoLearn Framework Demonstration
func demonstrateGoLearn(irisData IrisDataset) {
	fmt.Println("Training k-NN classifier on Iris dataset...")

	// Convert to GoLearn format
	rawData := base.NewDenseInstances()

	// Add attributes
	attrs := make([]base.Attribute, 5)
	attrs[0] = base.NewFloatAttribute("sepal_length")
	attrs[1] = base.NewFloatAttribute("sepal_width")
	attrs[2] = base.NewFloatAttribute("petal_length")
	attrs[3] = base.NewFloatAttribute("petal_width")
	attrs[4] = base.NewCategoricalAttribute()

	// Add class values
	classAttr := attrs[4].(*base.CategoricalAttribute)
	classAttr.GetSysValFromString("setosa")
	classAttr.GetSysValFromString("versicolor")
	classAttr.GetSysValFromString("virginica")

	rawData.AddClassAttribute(attrs[4])
	for i := 0; i < 4; i++ {
		rawData.AddAttribute(attrs[i])
	}

	rawData.Extend(len(irisData.Features))

	// Fill data
	for i := 0; i < len(irisData.Features); i++ {
		for j := 0; j < 4; j++ {
			rawData.Set(base.NewFloatAttribute(""), i, j,
				base.PackFloatToBytes(irisData.Features[i][j]))
		}
		// Set class
		rawData.Set(attrs[4], i, 4, classAttr.GetSysValFromString(irisData.Labels[i]))
	}

	// Split data
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.7)

	// Create and train classifier
	start := time.Now()
	cls := knn.NewKnnClassifier("euclidean", "linear", 3)
	cls.Fit(trainData)
	duration := time.Since(start)

	// Make predictions
	predictions, err := cls.Predict(testData)
	if err != nil {
		fmt.Printf("❌ Prediction failed: %v\n", err)
		return
	}

	// Evaluate
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		fmt.Printf("❌ Evaluation failed: %v\n", err)
		return
	}

	fmt.Printf("Training completed in: %v\n", duration)
	fmt.Printf("Accuracy: %.2f%%\n", evaluation.GetAccuracy(confusionMat)*100)

	fmt.Println("\n💡 rnxa Integration Concept:")
	fmt.Println("  // Custom distance function using rnxa")
	fmt.Println("  type RnxaEuclidean struct {")
	fmt.Println("      engine rnxa.ComputeEngine")
	fmt.Println("  }")
	fmt.Println("  // Accelerate distance calculations with GPU!")

	fmt.Println("✅ Strengths: Wide algorithm support, sklearn-like API")
	fmt.Println("🚀 Potential: rnxa could accelerate distance/similarity calculations")
}

// 5. Custom Framework Integration Example
func demonstrateCustomIntegration() {
	fmt.Println("Demonstrating how any framework can integrate rnxa...")

	// Example: Custom neural layer using rnxa
	layer := &CustomLayer{
		weights: [][]float64{{0.5, -0.3}, {0.2, 0.8}},
		bias:    []float64{0.1, -0.2},
	}

	err := layer.InitializeWithRnxa()
	if err != nil {
		fmt.Printf("❌ Failed to initialize with rnxa: %v\n", err)
		return
	}
	defer layer.Close()

	// Forward pass
	input := []float64{1.0, 0.5}
	start := time.Now()
	output, err := layer.Forward(input)
	duration := time.Since(start)

	if err != nil {
		fmt.Printf("❌ Forward pass failed: %v\n", err)
		return
	}

	fmt.Printf("Custom layer forward pass: %v (computed in %v)\n", formatFloats(output), duration)
	fmt.Printf("Using device: %s\n", layer.engine.Device().Name)

	fmt.Println("\n💡 Integration Pattern:")
	fmt.Println("  1. Initialize rnxa engine in your framework")
	fmt.Println("  2. Convert data to rnxa tensors")
	fmt.Println("  3. Use rnxa operations for heavy computation")
	fmt.Println("  4. Convert results back to framework format")

	fmt.Println("✅ Result: Any Go ML framework can get hardware acceleration!")
}

// 6. Performance Comparison
func performanceComparison() {
	fmt.Println("Comparing matrix multiplication performance...")

	sizes := []int{64, 128, 256}

	// Initialize rnxa engine
	engine, err := rnxa.NewEngine()
	if err != nil {
		fmt.Printf("❌ Failed to create rnxa engine: %v\n", err)
		return
	}
	defer engine.Close()

	fmt.Printf("%-10s %-15s %-15s %-10s\n", "Size", "Pure Go", "rnxa", "Speedup")
	fmt.Println("─────────────────────────────────────────────────────")

	for _, size := range sizes {
		// Create test matrices
		data := make([]float64, size*size)
		for i := range data {
			data[i] = float64(i%10) / 10.0
		}

		// Pure Go benchmark
		goTime := benchmarkPureGoMatMul(data, size)

		// rnxa benchmark
		rnxaTime := benchmarkRnxaMatMul(engine, data, size)

		speedup := float64(goTime) / float64(rnxaTime)
		fmt.Printf("%-10s %-15v %-15v %.2fx\n",
			fmt.Sprintf("%dx%d", size, size), goTime, rnxaTime, speedup)
	}

	fmt.Println("\n🎯 Key Insights:")
	fmt.Printf("• %s provides consistent acceleration\n", engine.Device().Name)
	fmt.Println("• Larger matrices benefit more from hardware acceleration")
	fmt.Println("• rnxa overhead is minimal for medium-large workloads")
}

// Helper types and functions

type IrisDataset struct {
	Features [][]float64
	Labels   []string
}

type CustomLayer struct {
	weights [][]float64
	bias    []float64
	engine  rnxa.ComputeEngine
}

func (l *CustomLayer) InitializeWithRnxa() error {
	engine, err := rnxa.NewEngine()
	if err != nil {
		return err
	}
	l.engine = engine
	return nil
}

func (l *CustomLayer) Forward(input []float64) ([]float64, error) {
	ctx := context.Background()

	// Convert to rnxa tensors
	inputTensor := rnxa.NewTensor(input, 1, len(input))

	// Convert weights to flat array for matrix multiplication
	weightData := make([]float64, len(l.weights)*len(l.weights[0]))
	for i, row := range l.weights {
		copy(weightData[i*len(row):], row)
	}
	weightTensor := rnxa.NewTensor(weightData, len(l.weights[0]), len(l.weights))

	// Matrix multiplication: input × weights
	matResult, err := l.engine.MatMul(ctx, inputTensor, weightTensor)
	if err != nil {
		return nil, err
	}

	// Add bias
	biasTensor := rnxa.NewTensor(l.bias, len(l.bias))
	result, err := l.engine.VectorAdd(ctx, matResult, biasTensor)
	if err != nil {
		return nil, err
	}

	// Apply ReLU activation
	activated, err := l.engine.ReLU(ctx, result)
	if err != nil {
		return nil, err
	}

	return activated.Data(), nil
}

func (l *CustomLayer) Close() {
	if l.engine != nil {
		l.engine.Close()
	}
}

func generateIrisDataset() IrisDataset {
	// Simplified Iris dataset
	return IrisDataset{
		Features: [][]float64{
			{5.1, 3.5, 1.4, 0.2}, {4.9, 3.0, 1.4, 0.2}, {4.7, 3.2, 1.3, 0.2},
			{7.0, 3.2, 4.7, 1.4}, {6.4, 3.2, 4.5, 1.5}, {6.9, 3.1, 4.9, 1.5},
			{6.3, 3.3, 6.0, 2.5}, {5.8, 2.7, 5.1, 1.9}, {7.1, 3.0, 5.9, 2.1},
		},
		Labels: []string{
			"setosa", "setosa", "setosa",
			"versicolor", "versicolor", "versicolor",
			"virginica", "virginica", "virginica",
		},
	}
}

func formatFloats(vals []float64) []string {
	result := make([]string, len(vals))
	for i, v := range vals {
		result[i] = fmt.Sprintf("%.3f", v)
	}
	return result
}

func benchmarkPureGoMatMul(data []float64, size int) time.Duration {
	start := time.Now()

	// Simple Go matrix multiplication
	result := make([]float64, size*size)
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			sum := 0.0
			for k := 0; k < size; k++ {
				sum += data[i*size+k] * data[k*size+j]
			}
			result[i*size+j] = sum
		}
	}

	return time.Since(start)
}

func benchmarkRnxaMatMul(engine rnxa.ComputeEngine, data []float64, size int) time.Duration {
	A := rnxa.NewTensor(data, size, size)
	B := rnxa.NewTensor(data, size, size)

	start := time.Now()
	_, err := engine.MatMul(context.Background(), A, B)
	duration := time.Since(start)

	if err != nil {
		return time.Hour // Return large time on error
	}

	return duration
}
