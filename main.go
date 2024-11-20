package main

import (
	"bufio"
	"bytes"
	"crypto/tls"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

const (
	openAIBaseURL = "https://vllm.atom.demo.supremind.info"
)

// UsageInfo holds the usage statistics from OpenAI API response
type UsageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatCompletionResponse represents a regular response
type ChatCompletionResponse struct {
	ID      string    `json:"id"`
	Object  string    `json:"object"`
	Created int64     `json:"created"`
	Model   string    `json:"model"`
	Usage   UsageInfo `json:"usage"`
}

// ChatCompletionStreamResponse represents a streaming response chunk
type ChatCompletionStreamResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *UsageInfo `json:"usage"`
}

// Metrics
var (
	promptTokens = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "openai_prompt_tokens_total",
			Help: "Total prompt tokens used from OpenAI API",
		},
		[]string{"model", "endpoint"},
	)
	completionTokens = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "openai_completion_tokens_total",
			Help: "Total completion tokens used from OpenAI API",
		},
		[]string{"model", "endpoint"},
	)
	totalTokens = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "openai_total_tokens_total",
			Help: "Total tokens used from OpenAI API",
		},
		[]string{"model", "endpoint"},
	)
	requestDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "openai_request_duration_seconds",
			Help:    "Duration of OpenAI API requests in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"endpoint", "status", "stream"},
	)
)

func init() {
	prometheus.MustRegister(promptTokens)
	prometheus.MustRegister(completionTokens)
	prometheus.MustRegister(totalTokens)
	prometheus.MustRegister(requestDuration)
}

type responseWriter struct {
	http.ResponseWriter
	model     string
	path      string
	start     time.Time
	isStream  bool
	buffer    *bytes.Buffer
	lastUsage *UsageInfo
	status    string
}

func newResponseWriter(w http.ResponseWriter) *responseWriter {
	return &responseWriter{
		ResponseWriter: w,
		start:          time.Now(),
		buffer:         &bytes.Buffer{},
	}
}

func (w *responseWriter) Write(b []byte) (int, error) {
	if !w.isStream {
		// For non-streaming responses, buffer the response to extract usage info
		w.buffer.Write(b)
		return w.ResponseWriter.Write(b)
	}

	// Handle SSE format for streaming responses
	scanner := bufio.NewScanner(bytes.NewReader(b))
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			if _, err := w.ResponseWriter.Write([]byte(line + "\n")); err != nil {
				return 0, err
			}
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			// Record final metrics when stream ends
			if w.lastUsage != nil {
				duration := time.Since(w.start).Seconds()
				recordMetrics(w.model, w.path, w.lastUsage, duration, "200", "true")
			}
			_, err := w.ResponseWriter.Write([]byte(line + "\n\n"))
			return len(b), err
		}

		var streamResp ChatCompletionStreamResponse
		if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
			log.Printf("Error unmarshaling stream response: %v", err)
			if _, err := w.ResponseWriter.Write([]byte(line + "\n\n")); err != nil {
				return 0, err
			}
			continue
		}

		// Update model if not set
		if w.model == "" && streamResp.Model != "" {
			w.model = streamResp.Model
		}

		// Update usage info from the last chunk that contains it
		if streamResp.Usage != nil {
			w.lastUsage = streamResp.Usage
		}

		// Write the original line back to client
		if _, err := w.ResponseWriter.Write([]byte(line + "\n\n")); err != nil {
			return 0, err
		}
	}

	return len(b), nil
}

func (w *responseWriter) recordNonStreamingMetrics() {
	if w.buffer.Len() == 0 {
		return
	}

	// Try to parse as regular response first
	var resp ChatCompletionResponse
	if err := json.Unmarshal(w.buffer.Bytes(), &resp); err != nil {
		log.Printf("Error unmarshaling non-stream response: %v", err)
		return
	}

	// Update model if found in response
	if resp.Model != "" {
		w.model = resp.Model
	}

	duration := time.Since(w.start).Seconds()
	recordMetrics(w.model, w.path, &resp.Usage, duration, "200", "false")
}

func recordMetrics(model, path string, usage *UsageInfo, duration float64, status, stream string) {
	if usage != nil {
		promptTokens.WithLabelValues(model, path).Add(float64(usage.PromptTokens))
		completionTokens.WithLabelValues(model, path).Add(float64(usage.CompletionTokens))
		totalTokens.WithLabelValues(model, path).Add(float64(usage.TotalTokens))
	}
	requestDuration.WithLabelValues(path, status, stream).Observe(duration)
}

// OpenAIProxy handles proxying requests to OpenAI and collecting metrics
func OpenAIProxy() http.Handler {
	target, _ := url.Parse(openAIBaseURL)
	proxy := httputil.NewSingleHostReverseProxy(target)

	// Configure transport to skip certificate verification
	proxy.Transport = &http.Transport{
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: true,
		},
	}

	// Modify the default director to handle the proxy logic
	defaultDirector := proxy.Director
	proxy.Director = func(req *http.Request) {
		defaultDirector(req)
		req.Host = target.Host

		// For chat completion requests, ensure we get usage statistics
		if strings.HasPrefix(req.URL.Path, "/v1/chat/completions") {
			var reqBody map[string]interface{}
			if req.Body != nil {
				bodyBytes, _ := io.ReadAll(req.Body)
				if err := json.Unmarshal(bodyBytes, &reqBody); err == nil {
					isStream, _ := reqBody["stream"].(bool)
					
					if isStream {
						// For streaming requests, set include_usage in stream_options
						streamOpts, ok := reqBody["stream_options"].(map[string]interface{})
						if !ok {
							streamOpts = make(map[string]interface{})
							reqBody["stream_options"] = streamOpts
						}
						streamOpts["include_usage"] = true
						
						modifiedBody, _ := json.Marshal(reqBody)
						req.Body = io.NopCloser(bytes.NewBuffer(modifiedBody))
						req.ContentLength = int64(len(modifiedBody))
					} else {
						// For non-streaming requests, don't modify the request
						req.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
					}
				} else {
					// Restore original body if unmarshal fails
					req.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
				}
			}
		}
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check if this is a streaming request by reading the request body
		isStream := false
		var model string
		if strings.HasPrefix(r.URL.Path, "/v1/chat/completions") {
			var reqBody map[string]interface{}
			if r.Body != nil {
				bodyBytes, _ := io.ReadAll(r.Body)
				if err := json.Unmarshal(bodyBytes, &reqBody); err == nil {
					isStream, _ = reqBody["stream"].(bool)
					if m, ok := reqBody["model"].(string); ok {
						model = m
					}
				}
				// Restore the body for further processing
				r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
			}
		}

		start := time.Now()
		path := r.URL.Path

		if isStream {
			// Handle streaming response
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")
			rw := newResponseWriter(w)
			rw.path = path
			rw.model = model
			rw.isStream = true
			proxy.ServeHTTP(rw, r)
			recordMetrics(rw.model, rw.path, rw.lastUsage, time.Since(start).Seconds(), "200", "true")
		} else {
			// Handle non-streaming response
			rw := newResponseWriter(w)
			rw.path = path
			proxy.ServeHTTP(rw, r)
			rw.recordNonStreamingMetrics()
			recordMetrics(rw.model, rw.path, nil, time.Since(start).Seconds(), "200", "false")
		}
	})
}

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	// Metrics endpoint
	http.Handle("/metrics", promhttp.Handler())
	
	// OpenAI proxy endpoint
	http.Handle("/v1/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Add OpenAI API key if not present
		if r.Header.Get("Authorization") == "" {
			r.Header.Set("Authorization", "Bearer "+apiKey)
		}
		OpenAIProxy().ServeHTTP(w, r)
	}))

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Starting OpenAI proxy server on :%s", port)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatalf("Error starting server: %v", err)
	}
}
