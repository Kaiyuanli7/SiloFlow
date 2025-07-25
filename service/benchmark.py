#!/usr/bin/env python3
"""
SiloFlow Pipeline Service Benchmark and Optimization Test Script
================================================================

This script tests the performance improvements of the optimized pipeline service.
"""

import asyncio
import aiohttp
import time
import json
import logging
from pathlib import Path
from typing import Dict, List
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineBenchmark:
    """Benchmark tool for testing pipeline performance."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    async def test_health_endpoint(self) -> Dict:
        """Test the health endpoint for basic connectivity."""
        logger.info("🔍 Testing health endpoint...")
        
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            
            try:
                async with session.get(f"{self.base_url}/health") as response:
                    response_time = time.time() - start_time
                    data = await response.json()
                    
                    return {
                        "endpoint": "/health",
                        "status_code": response.status,
                        "response_time": response_time,
                        "success": response.status == 200,
                        "data": data
                    }
            except Exception as e:
                return {
                    "endpoint": "/health",
                    "status_code": 0,
                    "response_time": time.time() - start_time,
                    "success": False,
                    "error": str(e)
                }
    
    async def test_metrics_endpoint(self) -> Dict:
        """Test the metrics endpoint for performance monitoring."""
        logger.info("📊 Testing metrics endpoint...")
        
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            
            try:
                async with session.get(f"{self.base_url}/metrics") as response:
                    response_time = time.time() - start_time
                    data = await response.json()
                    
                    return {
                        "endpoint": "/metrics",
                        "status_code": response.status,
                        "response_time": response_time,
                        "success": response.status == 200,
                        "data": data
                    }
            except Exception as e:
                return {
                    "endpoint": "/metrics",
                    "status_code": 0,
                    "response_time": time.time() - start_time,
                    "success": False,
                    "error": str(e)
                }
    
    async def create_test_data(self, filename: str = "test_data.csv", rows: int = 1000) -> Path:
        """Create test CSV data for pipeline testing."""
        import pandas as pd
        import numpy as np
        
        logger.info(f"📝 Creating test data: {filename} with {rows} rows")
        
        # Generate realistic grain temperature data
        np.random.seed(42)  # For reproducible results
        
        data = []
        granaries = ["Granary_A", "Granary_B", "Granary_C"]
        
        base_time = pd.Timestamp.now() - pd.Timedelta(days=30)
        
        for i in range(rows):
            granary = np.random.choice(granaries)
            timestamp = base_time + pd.Timedelta(hours=i * 0.1)
            
            # Simulate temperature with some variation
            base_temp = 25.0 + np.random.normal(0, 2.0)
            
            data.append({
                "granary_id": granary,
                "heap_id": f"Heap_{np.random.randint(1, 6)}",
                "detection_time": timestamp.isoformat(),
                "temperature_grain": base_temp,
                "grid_x": np.random.randint(1, 10),
                "grid_y": np.random.randint(1, 10),
                "grid_z": np.random.randint(1, 5),
            })
        
        df = pd.DataFrame(data)
        test_file = Path(filename)
        df.to_csv(test_file, index=False)
        
        logger.info(f"✅ Created test file: {test_file} ({test_file.stat().st_size / 1024:.1f} KB)")
        return test_file
    
    async def test_pipeline_endpoint(self, test_file: Path, concurrency: int = 1) -> List[Dict]:
        """Test the pipeline endpoint with different concurrency levels."""
        logger.info(f"🚀 Testing pipeline endpoint (concurrency: {concurrency})")
        
        results = []
        
        async def single_pipeline_test():
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                try:
                    # Prepare multipart form data
                    data = aiohttp.FormData()
                    data.add_field('file', open(test_file, 'rb'), filename=test_file.name)
                    data.add_field('horizon', '7')
                    
                    async with session.post(f"{self.base_url}/pipeline", data=data) as response:
                        response_time = time.time() - start_time
                        
                        # Read response
                        if response.content_type == 'text/csv':
                            content = await response.text()
                            content_size = len(content)
                        else:
                            content = await response.json()
                            content_size = len(str(content))
                        
                        return {
                            "status_code": response.status,
                            "response_time": response_time,
                            "content_size": content_size,
                            "success": response.status == 200,
                            "processing_time": response.headers.get("X-Processing-Time"),
                            "optimization_version": response.headers.get("X-Optimization-Version"),
                        }
                        
                except Exception as e:
                    return {
                        "status_code": 0,
                        "response_time": time.time() - start_time,
                        "content_size": 0,
                        "success": False,
                        "error": str(e)
                    }
        
        # Run concurrent tests
        tasks = [single_pipeline_test() for _ in range(concurrency)]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def run_benchmark_suite(self) -> Dict:
        """Run the complete benchmark suite."""
        logger.info("🏁 Starting SiloFlow Pipeline Benchmark Suite")
        start_time = time.time()
        
        benchmark_results = {
            "benchmark_start": start_time,
            "service_url": self.base_url,
            "tests": {}
        }
        
        # Test 1: Health endpoint
        health_result = await self.test_health_endpoint()
        benchmark_results["tests"]["health"] = health_result
        
        if not health_result["success"]:
            logger.error("❌ Health check failed - service may not be running")
            return benchmark_results
        
        # Test 2: Metrics endpoint
        metrics_result = await self.test_metrics_endpoint()
        benchmark_results["tests"]["metrics"] = metrics_result
        
        # Test 3: Create test data
        test_file = await self.create_test_data("benchmark_test.csv", 1000)
        
        # Test 4: Pipeline endpoint (single request)
        single_results = await self.test_pipeline_endpoint(test_file, concurrency=1)
        benchmark_results["tests"]["pipeline_single"] = single_results[0]
        
        # Test 5: Pipeline endpoint (concurrent requests)
        concurrent_results = await self.test_pipeline_endpoint(test_file, concurrency=3)
        benchmark_results["tests"]["pipeline_concurrent"] = {
            "concurrency": 3,
            "results": concurrent_results,
            "avg_response_time": statistics.mean([r["response_time"] for r in concurrent_results if r["success"]]),
            "success_rate": sum(1 for r in concurrent_results if r["success"]) / len(concurrent_results),
        }
        
        # Cleanup test file
        test_file.unlink()
        
        # Calculate overall results
        total_time = time.time() - start_time
        benchmark_results["benchmark_duration"] = total_time
        benchmark_results["summary"] = self._generate_summary(benchmark_results)
        
        logger.info(f"✅ Benchmark completed in {total_time:.2f}s")
        return benchmark_results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate a summary of benchmark results."""
        summary = {
            "service_status": "unknown",
            "optimization_detected": False,
            "performance_grade": "F",
            "recommendations": []
        }
        
        # Check service health
        if results["tests"]["health"]["success"]:
            summary["service_status"] = "healthy"
            health_data = results["tests"]["health"]["data"]
            
            # Check for optimization features
            if "optimizations" in health_data:
                optimizations = health_data["optimizations"]
                if optimizations.get("async_processing") and optimizations.get("performance_monitoring"):
                    summary["optimization_detected"] = True
        
        # Evaluate performance
        if "pipeline_single" in results["tests"] and results["tests"]["pipeline_single"]["success"]:
            response_time = results["tests"]["pipeline_single"]["response_time"]
            
            if response_time < 5:
                summary["performance_grade"] = "A"
            elif response_time < 10:
                summary["performance_grade"] = "B"
            elif response_time < 20:
                summary["performance_grade"] = "C"
            elif response_time < 30:
                summary["performance_grade"] = "D"
            else:
                summary["performance_grade"] = "F"
        
        # Generate recommendations
        if not summary["optimization_detected"]:
            summary["recommendations"].append("Enable optimization features in the service")
        
        if summary["performance_grade"] in ["D", "F"]:
            summary["recommendations"].append("Consider increasing system resources")
            summary["recommendations"].append("Enable concurrent processing")
        
        return summary
    
    def save_results(self, results: Dict, filename: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"💾 Results saved to {filename}")


async def main():
    """Main benchmark execution."""
    benchmark = PipelineBenchmark()
    
    logger.info("🚀 SiloFlow Pipeline Performance Benchmark")
    logger.info("=" * 50)
    
    try:
        results = await benchmark.run_benchmark_suite()
        
        # Print summary
        summary = results.get("summary", {})
        logger.info("\n📋 BENCHMARK SUMMARY:")
        logger.info(f"Service Status: {summary.get('service_status', 'unknown')}")
        logger.info(f"Optimizations Detected: {summary.get('optimization_detected', False)}")
        logger.info(f"Performance Grade: {summary.get('performance_grade', 'N/A')}")
        
        if summary.get("recommendations"):
            logger.info("💡 Recommendations:")
            for rec in summary["recommendations"]:
                logger.info(f"  - {rec}")
        
        # Save results
        benchmark.save_results(results)
        
        logger.info("\n✅ Benchmark completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
