# Multi-GPU Support Guide for SiloFlow

## 🎯 Overview

SiloFlow now includes comprehensive multi-GPU support with intelligent GPU selection, proper device management, and optimized training configurations. This guide explains how to effectively use multiple GPUs in your environment.

## 🔍 GPU Detection and Selection

### Automatic GPU Detection

SiloFlow automatically detects all available GPUs on your system and provides detailed information:

```bash
# Check GPU status when starting SiloFlow
python granary_pipeline.py --help
```

You'll see output like:
```
🔍 GPU Detection Summary: 2 GPU(s) found via pynvml
   GPU 0: NVIDIA GeForce RTX 4090 (24576MB, 23040MB free, 15% util)
   GPU 1: NVIDIA GeForce RTX 3080 (10240MB, 8192MB free, 45% util)
```

### GPU Selection Logic

**Auto-Selection Algorithm** (when no specific GPU is specified):
1. **Free Memory (40% weight)**: Prioritizes GPUs with more available memory
2. **GPU Utilization (30% weight)**: Prefers less utilized GPUs
3. **Total Memory (20% weight)**: Favors GPUs with larger total memory
4. **Memory Utilization (10% weight)**: Avoids heavily memory-loaded GPUs

**Manual Selection**: You can specify exactly which GPU to use with `--gpu-id`

## 🚀 Usage Examples

### Training with Automatic GPU Selection

```bash
# Enable GPU with auto-selection of best available GPU
python granary_pipeline.py train --granary "MyGranary" --gpu
```

### Training with Specific GPU Selection

```bash
# Use GPU 0 specifically
python granary_pipeline.py train --granary "MyGranary" --gpu --gpu-id 0

# Use GPU 1 specifically  
python granary_pipeline.py train --granary "MyGranary" --gpu --gpu-id 1
```

### Forecasting with Multi-GPU Support

```bash
# Auto-select best GPU for forecasting
python granary_pipeline.py forecast --granary "MyGranary" --gpu

# Use specific GPU for forecasting
python granary_pipeline.py forecast --granary "MyGranary" --gpu --gpu-id 1
```

## ⚙️ GPU Configuration Optimization

### Memory-Based Optimization

SiloFlow automatically adjusts LightGBM parameters based on:
- **Dataset size**: Number of rows in your dataset
- **Feature count**: Number of features being used
- **GPU memory**: Available memory on the selected GPU

### Automatic Parameter Adjustment

| Dataset + GPU Memory | max_bin Setting | Memory Usage |
|---------------------|----------------|--------------|
| Large dataset + Low GPU memory | 127 | Very Conservative |
| Medium dataset + Medium GPU memory | 255 | Conservative |
| Small dataset + High GPU memory | 511 | Standard |

### Example Optimizations

**For 500K+ row datasets on 8GB GPU**:
```
🎯 Using GPU 1 with max_bin=255 (conservative for large dataset)
```

**For 100K row datasets on 24GB GPU**:
```
🎯 Using GPU 0 with max_bin=511 (standard for adequate memory)
```

## 🛠️ Multi-GPU Environment Setup

### Prerequisites

1. **NVIDIA GPUs**: Install appropriate NVIDIA drivers
2. **Python packages**:
   ```bash
   pip install pynvml  # For NVIDIA GPU detection
   pip install pyopencl  # For OpenCL GPU detection (fallback)
   ```

### Environment Variables (Optional)

```bash
# Force specific GPU visibility (if needed)
export CUDA_VISIBLE_DEVICES=0,1  # Only show GPU 0 and 1
export CUDA_VISIBLE_DEVICES=1    # Only show GPU 1
```

## 📊 Performance Considerations

### When to Use Multiple GPUs

**Best Use Cases**:
- **Large datasets** (100K+ rows): GPU acceleration provides significant speedup
- **Multiple training jobs**: Use different GPUs for parallel granary training
- **Memory-intensive models**: Distribute across GPUs based on memory availability

**Not Recommended**:
- **Small datasets** (<10K rows): CPU might be faster due to GPU overhead
- **Single GPU systems**: Obviously limited to one GPU

### Performance Tips

1. **Monitor GPU utilization** during training
2. **Use larger batch sizes** when GPU memory allows
3. **Balance workload** across GPUs for multiple granaries
4. **Check memory usage** to avoid OOM errors

## 🔧 Troubleshooting

### GPU Not Detected

```bash
# Check if GPUs are visible to Python
python -c "import pynvml; pynvml.nvmlInit(); print(f'GPUs: {pynvml.nvmlDeviceGetCount()}')"
```

### Memory Issues

- **Reduce max_bin** manually in code if getting OOM errors
- **Use smaller dataset samples** for initial testing
- **Monitor GPU memory** with `nvidia-smi`

### Performance Issues

- **Check GPU utilization** with `nvidia-smi -l 1`
- **Ensure adequate system RAM** (2-4x dataset size)
- **Verify fast storage** (SSD recommended for large datasets)

## 📈 Expected Performance Improvements

### Training Speed

| Dataset Size | GPU vs CPU Speedup | Memory Requirements |
|-------------|-------------------|-------------------|
| 50K rows | 2-3x faster | 2-4GB GPU memory |
| 200K rows | 5-8x faster | 6-12GB GPU memory |
| 500K+ rows | 10-15x faster | 12-24GB GPU memory |

### Multi-GPU Scaling

| GPUs | Parallel Training | Memory Distribution |
|------|------------------|-------------------|
| 1 GPU | Single granary | Full dataset |
| 2 GPUs | 2 granaries | Split workload |
| 4 GPUs | 4 granaries | Distributed training |

## 🎛️ Configuration Examples

### High-Memory GPU Setup (24GB+ GPU)
```bash
# Use maximum performance settings
python granary_pipeline.py train --granary "LargeGranary" --gpu --gpu-id 0
# Will automatically use max_bin=511 for optimal performance
```

### Multi-GPU Production Setup
```bash
# Terminal 1: Train granary 1 on GPU 0
python granary_pipeline.py train --granary "Granary1" --gpu --gpu-id 0

# Terminal 2: Train granary 2 on GPU 1
python granary_pipeline.py train --granary "Granary2" --gpu --gpu-id 1
```

### Memory-Constrained Setup (8GB GPU)
```bash
# Will automatically use conservative settings
python granary_pipeline.py train --granary "MediumGranary" --gpu
# max_bin will be auto-adjusted based on dataset size
```

## 🔍 Monitoring and Debugging

### Check GPU Status
```bash
# Monitor GPU usage during training
nvidia-smi -l 1

# Check GPU memory details
nvidia-ml-py3 -q
```

### Logging Information

SiloFlow provides detailed GPU information in logs:
```
✅ GPU acceleration detected and enabled: NVIDIA GeForce RTX 4090 (24576MB, user-specified GPU 0)
🎯 GPU Configuration: Device 0, max_bin=511
🔍 GPU Detection Summary: 2 GPU(s) found via pynvml
   GPU 0: NVIDIA GeForce RTX 4090 (24576MB, 23040MB free, 15% util)
   GPU 1: NVIDIA GeForce RTX 3080 (10240MB, 8192MB free, 45% util)
```

## 🚨 Important Notes

1. **GPU selection is persistent** throughout the training session
2. **Memory optimization is automatic** but can be overridden in code
3. **Fallback to CPU** happens automatically if GPU setup fails
4. **Multi-GPU detection requires pynvml** (install with `pip install pynvml`)
5. **OpenCL fallback** available for non-NVIDIA GPUs

---

This comprehensive multi-GPU support ensures optimal performance and resource utilization in your SiloFlow deployments, whether you're training single models or running multiple granaries in parallel.
