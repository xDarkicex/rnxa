package compute

import (
	"context"
	"testing"

	"github.com/xDarkicex/rnxa"
)

// TestCPUPurego_AvailableWithoutLib verifies the off-heap-only contract:
// when librnxa_cpu is not installed, the engine reports Available()==false
// and methods return a clear "build the lib" error. The test is the
// primary regression guard against re-introducing a slow Go fallback.
func TestCPUPurego_AvailableWithoutLib(t *testing.T) {
	e := newCPUPuregoEngine()
	if e == nil {
		t.Fatal("newCPUPuregoEngine returned nil")
	}
	if e.Available() {
		t.Skip("librnxa_cpu appears to be installed; skipping unavailable-only test")
	}
	_, err := e.MatMul(context.Background(), nil, nil)
	if err == nil {
		t.Fatal("expected MatMul to error when the C++ lib is not loaded")
	}
}

// TestCPUPurego_LibNameSetPerPlatform ensures the build-tagged files
// each set cpuLibName. A blank cpuLibName would cause Dlopen to
// fail for any platform, which is a silent misconfiguration.
func TestCPUPurego_LibNameSetPerPlatform(t *testing.T) {
	if cpuLibName == "" {
		t.Fatalf("cpuLibName is empty for this platform; one of the build-tagged "+
			"cpu_purego_{darwin,linux,windows}.go files must set it in init()")
	}
}

// TestNewEngine_FailsLoudWhenNoBackend verifies the "fail-loud over
// slow-fallback" guarantee: with no usable backend the constructor
// returns a clear error rather than a half-working engine.
func TestNewEngine_FailsLoudWhenNoBackend(t *testing.T) {
	// If any backend is available (Metal on darwin, etc.), skip.
	if eng, err := rnxa.NewEngine(); err == nil && eng != nil && eng.Available() {
		_ = eng.Close()
		t.Skip("a real backend is available; this test only runs when none is")
	}
	_, err := rnxa.NewEngine()
	if err == nil {
		t.Fatal("NewEngine should return an error when no backend is available")
	}
}
