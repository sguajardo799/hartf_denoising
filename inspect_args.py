import inspect
from asteroid.losses import SingleSrcPMSQE, SingleSrcNegSTOI

print("SingleSrcPMSQE args:")
print(inspect.signature(SingleSrcPMSQE.__init__))

print("\nSingleSrcNegSTOI args:")
print(inspect.signature(SingleSrcNegSTOI.__init__))
