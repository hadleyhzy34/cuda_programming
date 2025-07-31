#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Represents a 3D point from a sensor
struct Point { float x, y, z; };

// --- Kernels ---

// 1. Preprocesses points, filtering any that are too close based on a dynamic threshold
__global__ void preprocess_points(Point* points, int n, float filter_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float dist_sq = points[idx].x * points[idx].x + points[idx].y * points[idx].y;
        if (dist_sq < filter_threshold * filter_threshold) {
            points[idx].z = -1.0f; // Mark as invalid
        }
    }
}

// 2. Analyzes points to detect potential hazards (simplified logic)
// In a real app, this would be a complex clustering algorithm (e.g., DBSCAN)
__global__ void detect_hazard(const Point* points, int n, int* out_hazard_flag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Simple logic: if a valid point is found very close, flag it as a hazard.
    // This is a reduction, but we simplify it by having each thread potentially write.
    if (idx < n && points[idx].z != -1.0f) {
        float dist_sq = points[idx].x * points[idx].x + points[idx].y * points[idx].y;
        // Check if a point is within a 2-meter "danger zone"
        if (dist_sq < 4.0f) {
            atomicExch(out_hazard_flag, 1);
        }
    }
}

// 3. (Conditional) A special kernel that runs ONLY if a hazard is detected
__global__ void log_hazard_kernel(int* hazard_log_counter) {
    // In a real app, this might copy hazard data to a special buffer.
    // Here, we just increment a counter.
    atomicAdd(hazard_log_counter, 1);
}


// --- Main Application Class ---

class SensorProcessor {
public:
    SensorProcessor(int points_per_frame);
    ~SensorProcessor();
    void process_frame(const std::vector<Point>& frame_data, float new_filter_threshold);

private:
    void build_graph();

    // Parameters
    int points_per_frame_;
    dim3 grid_, block_;

    // Host & Device Data
    Point* d_points_;
    int* d_hazard_flag_;
    int* d_hazard_log_counter_;
    int* h_hazard_log_counter_;

    // CUDA Graph objects
    cudaGraph_t main_graph_;
    cudaGraphExec_t instance_;
    cudaGraphNode_t preprocess_node_; // We need a handle to this node to update it
    cudaStream_t stream_;
};

SensorProcessor::SensorProcessor(int points_per_frame) : points_per_frame_(points_per_frame) {
    std::cout << "Initializing sensor processor..." << std::endl;

    // Setup kernel launch params
    block_ = dim3(256);
    grid_ = dim3((points_per_frame_ + block_.x - 1) / block_.x);

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_points_, points_per_frame_ * sizeof(Point)));
    CHECK_CUDA(cudaMalloc(&d_hazard_flag_, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_hazard_log_counter_, sizeof(int)));

    // Allocate page-locked host memory for async status checks
    CHECK_CUDA(cudaMallocHost(&h_hazard_log_counter_, sizeof(int)));
    *h_hazard_log_counter_ = 0;
    CHECK_CUDA(cudaMemset(d_hazard_log_counter_, 0, sizeof(int)));
    
    // Create stream for operations
    CHECK_CUDA(cudaStreamCreate(&stream_));

    // Build the graph structure and instantiate it
    build_graph();
    
    std::cout << "Initialization complete." << std::endl;
}

void SensorProcessor::build_graph() {
    std::cout << "Building CUDA graph..." << std::endl;

    // --- Create Main Graph and Nodes Manually ---
    CHECK_CUDA(cudaGraphCreate(&main_graph_, 0));

    // We need handles to chain dependencies
    cudaGraphNode_t memcpy_h2d_node, hazard_detect_node, conditional_node;
    
    // Node 1: Memcpy Host-to-Device (parameters will be updated at runtime)
    // We add it as an empty node here and will update its params later if needed,
    // or more commonly, the H2D copy happens outside the graph launch if data changes every frame.
    // For simplicity in this example, we'll do the memcpy outside the main graph launch.

    // Node 2: Preprocess Kernel (we save the handle to update its threshold)
    cudaKernelNodeParams preprocess_params = {0};
    preprocess_params.func = (void*)preprocess_points;
    preprocess_params.gridDim = grid_;
    preprocess_params.blockDim = block_;
    // Initial dummy params, will be updated before first launch
    float initial_threshold = 5.0f;
    void* preprocess_args[] = {(void*)&d_points_, (void*)&points_per_frame_, (void*)&initial_threshold};
    preprocess_params.kernelParams = preprocess_args;
    CHECK_CUDA(cudaGraphAddKernelNode(&preprocess_node_, main_graph_, nullptr, 0, &preprocess_params));

    // Node 3: Hazard Detection Kernel
    cudaKernelNodeParams detect_params = {0};
    detect_params.func = (void*)detect_hazard;
    detect_params.gridDim = grid_;
    detect_params.blockDim = block_;
    void* detect_args[] = {(void*)&d_points_, (void*)&points_per_frame_, (void*)&d_hazard_flag_};
    detect_params.kernelParams = detect_args;
    CHECK_CUDA(cudaGraphAddKernelNode(&hazard_detect_node, main_graph_, &preprocess_node_, 1, &detect_params));

    // --- Setup for Conditional Execution ---
    cudaGraph_t graph_if_hazard, graph_if_no_hazard;
    cudaGraphNode_t log_node;

    // Create the subgraph that runs if a hazard is found
    CHECK_CUDA(cudaGraphCreate(&graph_if_hazard, 0));
    cudaKernelNodeParams log_params = {0};
    log_params.func = (void*)log_hazard_kernel;
    log_params.gridDim = dim3(1); log_params.blockDim = dim3(1);
    void* log_args[] = {(void*)&d_hazard_log_counter_};
    log_params.kernelParams = log_args;
    CHECK_CUDA(cudaGraphAddKernelNode(&log_node, graph_if_hazard, nullptr, 0, &log_params));

    // Create the empty "else" subgraph
    CHECK_CUDA(cudaGraphCreate(&graph_if_no_hazard, 0));
    
    // Create a handle to the GPU variable that will be checked
    cudaGraphConditionalHandle cond_handle;
    CHECK_CUDA(cudaGraphConditionalHandleCreate(&cond_handle, d_hazard_flag_, 1, cudaGraphConditionalDefault));

    // Node 4: The Conditional Node itself
    cudaGraphNodeParams_conditional conditional_params = {};
    conditional_params.handle = cond_handle;
    conditional_params.if_graph = graph_if_hazard;
    conditional_params.else_graph = graph_if_no_hazard;
    conditional_params.type = cudaGraphNodeTypeConditional;
    CHECK_CUDA(cudaGraphAddNode(&conditional_node, main_graph_, &hazard_detect_node, 1, &conditional_params));
    
    // --- Instantiate the final graph ---
    CHECK_CUDA(cudaGraphInstantiate(&instance_, main_graph_, NULL, NULL, 0));
    std::cout << "Graph built and instantiated." << std::endl;
}

void SensorProcessor::process_frame(const std::vector<Point>& frame_data, float new_filter_threshold) {
    // Reset the hazard flag on the GPU before processing
    CHECK_CUDA(cudaMemsetAsync(d_hazard_flag_, 0, sizeof(int), stream_));

    // 1. Asynchronously copy new frame data to the device
    CHECK_CUDA(cudaMemcpyAsync(d_points_, frame_data.data(), frame_data.size() * sizeof(Point), cudaMemcpyHostToDevice, stream_));

    // --- GRAPH UPDATE ---
    // Update the filter threshold in our instantiated graph. This is very fast.
    cudaKernelNodeParams preprocess_params = {0};
    preprocess_params.func = (void*)preprocess_points;
    preprocess_params.gridDim = grid_;
    preprocess_params.blockDim = block_;
    void* preprocess_args[] = {(void*)&d_points_, (void*)&points_per_frame_, (void*)&new_filter_threshold};
    preprocess_params.kernelParams = preprocess_args;
    CHECK_CUDA(cudaGraphExecKernelNodeSetParams(instance_, preprocess_node_, &preprocess_params));

    // --- GRAPH LAUNCH ---
    // Launch the entire updated pipeline. The GPU will handle the conditional logic internally.
    CHECK_CUDA(cudaGraphLaunch(instance_, stream_));

    // Wait for this frame to be fully processed before continuing
    CHECK_CUDA(cudaStreamSynchronize(stream_));
}

SensorProcessor::~SensorProcessor() {
    std::cout << "Cleaning up..." << std::endl;
    CHECK_CUDA(cudaGraphExecDestroy(instance_));
    CHECK_CUDA(cudaGraphDestroy(main_graph_));
    // The conditional subgraphs are owned by the main graph, no need to destroy them separately
    CHECK_CUDA(cudaStreamDestroy(stream_));
    CHECK_CUDA(cudaFree(d_points_));
    CHECK_CUDA(cudaFree(d_hazard_flag_));
    CHECK_CUDA(cudaFree(d_hazard_log_counter_));
    CHECK_CUDA(cudaFreeHost(h_hazard_log_counter_));
}


int main() {
    const int points_per_frame = 1024 * 128; // 128k points
    SensorProcessor processor(points_per_frame);

    // Simulate a stream of 10 sensor frames
    for (int i = 0; i < 10; ++i) {
        std::cout << "\n--- Processing Frame " << i << " ---" << std::endl;

        // Generate some random point cloud data for this frame
        std::vector<Point> frame_data(points_per_frame);
        std::mt19937 gen(i); // Seed with frame number for variety
        std::uniform_real_distribution<> dis(-10.0, 10.0);
        for(auto& p : frame_data) { p.x = dis(gen); p.y = dis(gen); p.z = 0; }
        // Manually insert a "hazard" point in some frames
        if (i == 3 || i == 7) {
            std::cout << "  (Injecting a hazard point into this frame)" << std::endl;
            frame_data[100].x = 1.0f; frame_data[100].y = 1.0f;
        }

        // Simulate adapting to the environment by changing the filter
        float filter_threshold = 5.0f - 0.2f * i;
        std::cout << "  Updating filter threshold to: " << filter_threshold << std::endl;

        processor.process_frame(frame_data, filter_threshold);
    }
    
    return 0;
}