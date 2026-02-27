import pandas as pd
from features.pipeline import FeaturePipeline
from features.technical.sma import SMAFeature
from core.monkeys.trend_monkey import TrendMonkey
from core.orchestrator import MarketOrchestrator

# 1. Create Mock Data
df = pd.DataFrame({
    "close": [10, 11, 12, 13, 14, 15, 14, 13, 12, 11]
})

# 2. Build Pipeline
pipeline = FeaturePipeline()
pipeline.add_feature(SMAFeature(window=2))
pipeline.add_feature(SMAFeature(window=3))

processed_df = pipeline.generate(df)
print("Pipeline Features Config:", pipeline.get_feature_name())
print("Processed Data Tail:\n", processed_df.tail(3))

# 3. Initialize Monkeys
fast_sma = pipeline.get_feature_name()[0]
slow_sma = pipeline.get_feature_name()[1]
trend_monkey = TrendMonkey("TrendBot", fast_col=fast_sma, slow_col=slow_sma)

# 4. Orchestrate
orchestrator = MarketOrchestrator([trend_monkey], activation_threshold=0.2)
consensus = orchestrator.get_consensus(processed_df)

print("\nNotion Dashboard Output:")
print(consensus)
