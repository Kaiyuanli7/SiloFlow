#!/usr/bin/env python3

from granarypredict.multi_lgbm import MultiLGBMRegressor
import pandas as pd
import numpy as np

print('🧪 Testing GPU detection caching fix...')

# Create test data
X = pd.DataFrame(np.random.randn(200, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
Y = pd.DataFrame(np.random.randn(200, 3), columns=['h1', 'h2', 'h3'])

# Test training with GPU - should only see GPU detection once during initialization
print('🚀 Creating MultiLGBMRegressor with GPU enabled...')
model = MultiLGBMRegressor(use_gpu=True, gpu_optimization=True)

print('\n🏋️ Starting training (GPU detection should only happen once during initialization)...')
model.fit(X, Y, verbose=True)

print('\n✅ Training completed! If you only saw one GPU detection sequence, the fix worked!')
